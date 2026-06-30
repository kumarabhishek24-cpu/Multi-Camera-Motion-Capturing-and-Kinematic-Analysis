import argparse
import os
from datetime import datetime
import threading

import cv2
import numpy as np
from openpyxl import Workbook

# ─────────────────────────────────────────────────────────────────
#  QUAD CAMERA LED TRACKER  (fixed + smooth)
#  python scripts/quad_led_tracking.py \
#    --cam-laptop 0 --cam1 1 --cam2 2 --cam-depth 4 \
#    --calibration calibration/stereo_calibration.npz
# ─────────────────────────────────────────────────────────────────

BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
for _n in ["realsense", "openni", "openni2"]:
    if hasattr(cv2, f"CAP_{_n.upper()}"):
        BACKENDS[_n] = getattr(cv2, f"CAP_{_n.upper()}")


def open_cam(index, backend="auto", width=1280, height=720, warmup=20, timeout=12.0):
    """
    FIX 1: increased warmup frames (15→20) and timeout (7→12s).
    Some cameras (especially MSMF depth cams) need extra time to stabilise.
    """
    order = (
        [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]
        if backend == "auto"
        else [(backend, BACKENDS.get(backend, cv2.CAP_ANY))]
    )
    result = {"cap": None, "name": None}

    def _try():
        for name, api in order:
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            # FIX 2: keep reading until we get a real frame, not just ret=True
            for _ in range(warmup):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    result["cap"]  = cap
                    result["name"] = name
                    return
            cap.release()

    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout)
    return result["cap"], result["name"]


def load_calibration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration not found: {path}")
    d  = np.load(path)
    K1 = d["cameraMatrix1"] if "cameraMatrix1" in d else d["K1"]
    D1 = d["distCoeffs1"]   if "distCoeffs1"   in d else d["D1"]
    K2 = d["cameraMatrix2"] if "cameraMatrix2" in d else d["K2"]
    D2 = d["distCoeffs2"]   if "distCoeffs2"   in d else d["D2"]
    return K1, D1, K2, D2, d["R"], d["T"]


# ── DETECTION ─────────────────────────────────────────────────────

def detect_led(frame, thresh, min_area=20):
    """
    Finds the phone LED torch — a small, intensely bright WHITE circle.

    Strategy
    ─────────
    1. Grayscale → find the single brightest pixel location (minMaxLoc).
    2. Flood-fill a small region around it to get the bloom extent.
    3. Sanity-check: blob must be roughly circular and not too large.
    4. If the brightest region fails the shape test, fall back to contour
       scanning with a scoring function that rewards:
         • high mean brightness inside the blob   (phone LED is ~255)
         • compact/circular shape                  (aspect ratio close to 1)
         • small-to-medium size                    (not a ceiling tube)

    Returns (cx, cy) as floats or None, plus a grayscale debug image.
    """
    if frame is None or frame.size == 0:
        return None, np.zeros((10, 10), np.uint8)

    h, w = frame.shape[:2]
    frame_area = h * w
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── FAST PATH: brightest pixel → flood-fill bloom ──────────────
    _, max_val, _, max_loc = cv2.minMaxLoc(gray)

    # If the brightest pixel isn't even close to saturated, nothing to track
    if max_val >= thresh:
        # Threshold tightly around that brightness level
        _, bright_mask = cv2.threshold(gray, max(thresh, max_val - 15),
                                       255, cv2.THRESH_BINARY)
        # Clean noise
        k3 = np.ones((3, 3), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, k3)
        # Merge bloom fragments
        k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        merged = cv2.dilate(bright_mask, k11, iterations=2)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Score every candidate; pick the BEST (not just smallest)
        best_score = -1.0
        best_cnt   = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            # Reject blobs > 8 % of frame (ceiling, window, monitor glow)
            if area > 0.08 * frame_area:
                continue
            bx, by, bw2, bh2 = cv2.boundingRect(c)
            aspect = max(bw2, bh2) / max(min(bw2, bh2), 1)
            # Reject long thin blobs (fluorescent tubes: aspect >> 1)
            if aspect > 3.5:
                continue

            # Mean brightness of original gray inside this contour
            blob_mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(blob_mask, [c], -1, 255, -1)
            mean_bright = cv2.mean(gray, mask=blob_mask)[0]

            # Circularity: 4π·area / perimeter²  → 1 = perfect circle
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * area / (perimeter ** 2 + 1e-6))

            # Score = brightness × circularity / aspect_penalty
            #   → rewards bright, round, compact blobs
            score = mean_bright * circularity / aspect
            if score > best_score:
                best_score = score
                best_cnt   = c

        debug_img = merged  # show the dilated mask for debugging

        if best_cnt is not None:
            m = cv2.moments(best_cnt)
            if m["m00"] > 0:
                cx = float(m["m10"] / m["m00"])
                cy = float(m["m01"] / m["m00"])
                return (cx, cy), debug_img

    # ── FALLBACK: lower threshold by 20 and retry once ─────────────
    lower = max(thresh - 20, 80)
    _, mask2 = cv2.threshold(gray, lower, 255, cv2.THRESH_BINARY)
    k3  = np.ones((3, 3), np.uint8)
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask2   = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, k3)
    merged2 = cv2.dilate(mask2, k11, iterations=2)
    contours2, _ = cv2.findContours(merged2, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    best_score2 = -1.0
    best_cnt2   = None
    for c in contours2:
        area = cv2.contourArea(c)
        if area < min_area or area > 0.08 * frame_area:
            continue
        bx, by, bw2, bh2 = cv2.boundingRect(c)
        aspect = max(bw2, bh2) / max(min(bw2, bh2), 1)
        if aspect > 3.5:
            continue
        blob_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(blob_mask, [c], -1, 255, -1)
        mean_bright  = cv2.mean(gray, mask=blob_mask)[0]
        perimeter    = cv2.arcLength(c, True)
        circularity  = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
        score        = mean_bright * circularity / aspect
        if score > best_score2:
            best_score2 = score
            best_cnt2   = c

    if best_cnt2 is not None:
        m = cv2.moments(best_cnt2)
        if m["m00"] > 0:
            return (float(m["m10"] / m["m00"]),
                    float(m["m01"] / m["m00"])), merged2

    return None, merged2 if 'merged2' in dir() else np.zeros((h, w), np.uint8)


# ── TRIANGULATION ─────────────────────────────────────────────────

def triangulate(pt1, pt2, K1, D1, K2, D2, R, T):
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, T))
    p1 = cv2.undistortPoints(
        np.array([[[pt1[0], pt1[1]]]], np.float64), K1, D1
    )
    p2 = cv2.undistortPoints(
        np.array([[[pt2[0], pt2[1]]]], np.float64), K2, D2
    )
    Xh = cv2.triangulatePoints(P1, P2, p1.reshape(2, 1), p2.reshape(2, 1))
    # FIX 6: Xh shape is (4,1) — extract scalar with [3, 0]
    w_coord = float(Xh[3, 0])
    if abs(w_coord) < 1e-9:
        return None
    return (Xh[:3, 0] / w_coord).astype(np.float64)


# ── DRAW HELPERS ──────────────────────────────────────────────────

# Global frame counter for pulsing animation
_frame_counter = 0

def draw_led(frame, pt, color):
    """
    Draws a large, highly visible LED marker:
      • soft white glow halo behind the colour ring
      • thick outer ring (radius 40) + thin inner ring (radius 18)
      • filled centre dot (radius 6)
      • crosshair lines extending 60 px each side
      • pulsing outer ring that grows/shrinks slightly each frame
    """
    global _frame_counter
    if pt is None:
        return
    cx, cy = int(round(pt[0])), int(round(pt[1]))

    # Pulsing: outer ring oscillates ±6 px at ~1 Hz (assumes ~30 fps)
    pulse = int(6 * np.sin(_frame_counter * 0.2))
    R_outer = 40 + pulse
    R_inner = 18
    LINE_LEN = 65
    WHITE = (255, 255, 255)

    # 1) White glow halo (slightly bigger, semi-transparent feel via thick white ring)
    cv2.circle(frame, (cx, cy), R_outer + 6, WHITE, 3)

    # 2) Thick coloured outer ring
    cv2.circle(frame, (cx, cy), R_outer, color, 3)

    # 3) Inner ring
    cv2.circle(frame, (cx, cy), R_inner, color, 2)

    # 4) Filled centre dot
    cv2.circle(frame, (cx, cy), 6, color, -1)
    cv2.circle(frame, (cx, cy), 6, WHITE, 1)   # white border on dot

    # 5) Crosshair lines (gap in centre so centre dot is visible)
    gap = R_inner + 2
    cv2.line(frame, (cx - LINE_LEN, cy), (cx - gap, cy),   color, 2)
    cv2.line(frame, (cx + gap,      cy), (cx + LINE_LEN, cy), color, 2)
    cv2.line(frame, (cx, cy - LINE_LEN), (cx, cy - gap),   color, 2)
    cv2.line(frame, (cx, cy + gap),      (cx, cy + LINE_LEN), color, 2)

    _frame_counter += 1


# ── LOGGER ────────────────────────────────────────────────────────

class Logger:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(out_dir, f"4cam_log_{ts}.xlsx")
        self.wb   = Workbook()
        self.ws   = self.wb.active
        self.ws.title = "LED Tracking"
        self.ws.append([
            "sample", "timestamp",
            "laptop_px_x", "laptop_px_y",
            "cam1_px_x",   "cam1_px_y",
            "cam2_px_x",   "cam2_px_y",
            "depth_px_x",  "depth_px_y",
            "X_raw", "Y_raw", "Z_raw",
            "X_smooth", "Y_smooth", "Z_smooth",
        ])
        self._n    = 0
        self._last = 0

    def add(self, pts, xyz_raw, xyz_sm):
        def px(pt, i):
            return round(float(pt[i]), 1) if pt is not None else None

        # FIX 7: numpy array truth-check was crashing — use explicit None check
        def xyz_val(arr, i):
            if arr is None:
                return None
            try:
                return round(float(arr[i]), 4)
            except Exception:
                return None

        self.ws.append([
            self._n,
            datetime.now().isoformat(timespec="milliseconds"),
            px(pts[0], 0), px(pts[0], 1),
            px(pts[1], 0), px(pts[1], 1),
            px(pts[2], 0), px(pts[2], 1),
            px(pts[3], 0), px(pts[3], 1),
            xyz_val(xyz_raw, 0), xyz_val(xyz_raw, 1), xyz_val(xyz_raw, 2),
            xyz_val(xyz_sm,  0), xyz_val(xyz_sm,  1), xyz_val(xyz_sm,  2),
        ])
        self._n += 1
        if self._n - self._last >= 30:
            self.wb.save(self.path)
            self._last = self._n

    def save(self):
        self.wb.save(self.path)
        print(f"Saved → {self.path}")


# ── MAIN ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-laptop",    type=int,   default=0)
    ap.add_argument("--cam1",          type=int,   default=1)
    ap.add_argument("--cam2",          type=int,   default=2)
    ap.add_argument("--cam-depth",     type=int,   default=4)
    ap.add_argument("--backend",       default="auto",
                    choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--depth-backend", default="msmf",
                    choices=["auto", "dshow", "msmf", "any",
                             "realsense", "openni", "openni2"])
    ap.add_argument("--width",         type=int,   default=1280)
    ap.add_argument("--height",        type=int,   default=720)
    ap.add_argument("--threshold",     type=int,   default=220,
                    help="Per-channel brightness threshold (lower=more sensitive).")
    ap.add_argument("--smooth",        type=float, default=0.3,
                    help="EMA alpha: 0=frozen, 1=no smoothing. Default 0.3.")
    ap.add_argument("--calibration",   default="calibration/stereo_calibration.npz")
    ap.add_argument("--output-dir",    default="outputs")
    args = ap.parse_args()

    script_dir   = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    def resolve(p):
        if os.path.isabs(p):
            return p
        if os.path.exists(os.path.abspath(p)):
            return os.path.abspath(p)
        return os.path.join(project_root, p)

    cal_path = resolve(args.calibration)
    session  = os.path.join(
        resolve(args.output_dir),
        "4cam_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    K1, D1, K2, D2, R, T = load_calibration(cal_path)
    logger = Logger(session)

    cam_cfgs = [
        ("Laptop", args.cam_laptop, args.backend),
        ("Cam1-L", args.cam1,       args.backend),
        ("Cam2-R", args.cam2,       args.backend),
        ("Depth",  args.cam_depth,  args.depth_backend),
    ]
    COLORS = [
        (0, 255, 255),   # yellow  – Laptop
        (0, 255, 0),     # green   – Cam1-L
        (0, 165, 255),   # orange  – Cam2-R
        (255, 0, 255),   # magenta – Depth
    ]

    print("Opening cameras…  (may take up to 12 s each)")
    caps = []
    for label, idx, bk in cam_cfgs:
        cap, used = open_cam(idx, bk, args.width, args.height)
        status = f"OK  backend={used}" if cap else "FAILED - will show blank"
        print(f"  {label:10s}  idx={idx}  {status}")
        caps.append(cap)

    # FIX 8: only HARD-fail if the stereo pair (cam1+cam2) is missing
    if caps[1] is None or caps[2] is None:
        print("\nERROR: Cam1-L and Cam2-R must both open for 3-D tracking.")
        for c in caps:
            if c:
                c.release()
        return

    # FIX 9: blank frame uses FULL resolution (not half) — sized at read time
    def make_blank(cap):
        if cap is None:
            return np.zeros((args.height, args.width, 3), np.uint8)
        return np.zeros((args.height, args.width, 3), np.uint8)

    WIN  = [f"cam_{l}"  for l, _, _ in cam_cfgs]
    BINW = [f"bin_{l}"  for l, _, _ in cam_cfgs]
    DISP_W, DISP_H = args.width // 2, args.height // 2

    for w in WIN + BINW:
        cv2.namedWindow(w, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(w, DISP_W, DISP_H)

    # One shared threshold trackbar on Cam1-L window
    cv2.createTrackbar("threshold", WIN[1], args.threshold, 255, lambda _: None)

    print(f"\nCalibration : {cal_path}")
    print(f"Output      : {session}")
    print("Threshold trackbar is on the Cam1-L window.")
    print("  Lower  → more sensitive   (try 200 if LED not detected)")
    print("  Higher → fewer false hits (try 240 if ceiling light triggers)")
    print("Press Q to quit.\n")

    xyz_smooth = None        # numpy float64 array or None

    # Per-camera pixel EMA state  — smooths the crosshair position on screen
    px_smooth  = [None] * 4  # each entry: [x, y] float or None
    px_lost    = [0]    * 4  # frame counter since last detection
    PX_ALPHA   = 0.30        # lower = smoother / more lag  (0.2 – 0.5 good)
    PX_HOLD    = 20          # frames to keep showing last position after loss

    try:
        while True:
            # ── Grab frames ──
            frames = []
            for cap in caps:
                if cap is not None:
                    ret, f = cap.read()
                    if ret and f is not None and f.size > 0:
                        frames.append(f)
                    else:
                        frames.append(make_blank(cap))
                else:
                    frames.append(make_blank(None))

            thresh = cv2.getTrackbarPos("threshold", WIN[1])
            thresh = max(thresh, 50)

            # ── Detect ──
            pts, bins = [], []
            for f in frames:
                pt, bw = detect_led(f, thresh)
                pts.append(pt)
                bins.append(bw)

            # ── Per-camera pixel EMA smoothing ──
            smooth_pts = []
            for i, pt in enumerate(pts):
                if pt is not None:
                    px_lost[i] = 0
                    if px_smooth[i] is None:
                        px_smooth[i] = [float(pt[0]), float(pt[1])]
                    else:
                        px_smooth[i][0] = PX_ALPHA * pt[0] + (1 - PX_ALPHA) * px_smooth[i][0]
                        px_smooth[i][1] = PX_ALPHA * pt[1] + (1 - PX_ALPHA) * px_smooth[i][1]
                    smooth_pts.append((px_smooth[i][0], px_smooth[i][1]))
                else:
                    px_lost[i] += 1
                    # Hold last known position for PX_HOLD frames, then drop
                    if px_smooth[i] is not None and px_lost[i] <= PX_HOLD:
                        smooth_pts.append((px_smooth[i][0], px_smooth[i][1]))
                    else:
                        px_smooth[i] = None
                        smooth_pts.append(None)

            # ── 3-D triangulation (uses smoothed pixel positions) ──
            xyz_raw = None
            if smooth_pts[1] is not None and smooth_pts[2] is not None:
                xyz_raw = triangulate(smooth_pts[1], smooth_pts[2],
                                      K1, D1, K2, D2, R, T)
                if xyz_raw is not None:
                    if xyz_smooth is None:
                        xyz_smooth = xyz_raw.copy()
                    else:
                        xyz_smooth = (args.smooth * xyz_raw
                                      + (1.0 - args.smooth) * xyz_smooth)

            if xyz_smooth is not None:
                xyz_txt = (f"X={xyz_smooth[0]: .3f}  "
                           f"Y={xyz_smooth[1]: .3f}  "
                           f"Z={xyz_smooth[2]: .3f} m")
            else:
                xyz_txt = "XYZ: need LED in both Cam1-L + Cam2-R"

            # ── Draw each camera window ──
            for i, (f, (label, idx, _), color) in enumerate(
                    zip(frames, cam_cfgs, COLORS)):

                # Draw on SMOOTH position (not raw detection)
                draw_led(f, smooth_pts[i], color)

                led_txt = (f"LED ({smooth_pts[i][0]:.0f}, {smooth_pts[i][1]:.0f})"
                           if smooth_pts[i] is not None else "LED: not found")

                cv2.putText(f, led_txt, (15,  35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, color,        2)
                cv2.putText(f, xyz_txt, (15,  70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255),  2)
                cv2.putText(f, f"{label}  idx={idx}", (15, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                cv2.imshow(WIN[i], cv2.resize(f, (DISP_W, DISP_H)))
                if bins[i] is not None:
                    cv2.imshow(BINW[i], cv2.resize(bins[i], (DISP_W, DISP_H)))

            logger.add(pts, xyz_raw, xyz_smooth)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        for cap in caps:
            if cap:
                cap.release()
        cv2.destroyAllWindows()
        logger.save()


if __name__ == "__main__":
    main()

# python ".\scripts\quad_led_tracking.py"
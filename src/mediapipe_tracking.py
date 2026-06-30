"""
mediapipe_tracking.py  —  Full-body pose tracking on the main (Laptop) cam
Compatible with mediapipe 0.10.30+  (new Tasks API)

First run will auto-download the pose_landmarker model (~6 MB) to:
  models/pose_landmarker_full.task

Usage:
    python scripts/mediapipe_tracking.py

Controls:
    Q  →  quit + save CSV & plot
    P  →  pause / resume
    S  →  save snapshot of current frame
"""

import cv2
import mediapipe as mp
import csv
import os
import sys
import time
import threading
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import tasks

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 0
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS    = 30

# Script lives in  …/scripts/  so project root is one level up
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

OUTPUT_BASE   = os.path.join(_PROJECT_ROOT, "outputs", "mediapipe")
MODEL_DIR     = os.path.join(_PROJECT_ROOT, "models")
MODEL_PATH    = os.path.join(MODEL_DIR, "pose_landmarker_full.task")
MODEL_URL     = ("https://storage.googleapis.com/mediapipe-models/"
                 "pose_landmarker/pose_landmarker_full/float16/latest/"
                 "pose_landmarker_full.task")

# Landmark indices (new API uses integer indices)
PL = mp_vision.PoseLandmark
TRACKED = {
    "right_wrist":    PL.RIGHT_WRIST.value,
    "left_wrist":     PL.LEFT_WRIST.value,
    "right_elbow":    PL.RIGHT_ELBOW.value,
    "left_elbow":     PL.LEFT_ELBOW.value,
    "right_shoulder": PL.RIGHT_SHOULDER.value,
    "left_shoulder":  PL.LEFT_SHOULDER.value,
}

VISIBILITY_THRESH = 0.5

# ── BlazePose visual style (green dots + red lines, same as mp.solutions.pose) ──
POSE_CONNECTIONS = [
    # Face
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    # Arms
    (11,12),
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    # Torso
    (11,23),(12,24),(23,24),
    # Legs
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]
JOINT_COLOR = (0,   255,   0)   # bright green dots  (BlazePose style)
BONE_COLOR  = (0,   0,   255)   # red lines          (BlazePose style)
TRACK_COLOR = (0,   255, 255)   # cyan ring on specifically tracked joints
DOT_RADIUS  = 5
DOT_BORDER  = 2                 # white border on dots


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────
def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading pose model (~6 MB) → {MODEL_PATH}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH,
                reporthook=lambda b, bs, ts: print(
                    f"  {min(b*bs, ts)*100//ts if ts>0 else 0}%", end="\r"))
            print("\n  Download complete.")
        except Exception as e:
            print(f"\nERROR downloading model: {e}")
            print(f"Download manually from:\n  {MODEL_URL}")
            print(f"Save to: {MODEL_PATH}")
            sys.exit(1)


def open_cam(index, width, height, fps, warmup=20, timeout=12.0):
    order  = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]
    result = {"cap": None, "name": None}

    def _try():
        for name, api in order:
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                cap.release(); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            for _ in range(warmup):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    result["cap"]  = cap
                    result["name"] = name
                    return
            cap.release()

    t = threading.Thread(target=_try, daemon=True)
    t.start(); t.join(timeout)
    return result["cap"], result["name"]


def save_csv(path, rows, landmark_names):
    header = ["timestamp_s"]
    for name in landmark_names:
        header += [f"{name}_x_px", f"{name}_y_px",
                   f"{name}_z_norm", f"{name}_visibility"]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(f"  CSV  → {path}")


def save_plot(path, data_by_landmark):
    names = [n for n, d in data_by_landmark.items() if d]
    if not names:
        print("  Plot skipped (no data).")
        return
    fig, axes = plt.subplots(len(names), 1,
                             figsize=(12, 4 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        series = data_by_landmark[name]
        ts = [r[3] for r in series]
        xs = [r[0] for r in series]
        ys = [r[1] for r in series]
        ax.plot(ts, xs, label="X (px)", color="royalblue",  linewidth=1.8)
        ax.plot(ts, ys, label="Y (px)", color="darkorange", linewidth=1.8)
        ax.set_title(name.replace("_", " ").title(),
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Pixel")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Elapsed time (s)")
    fig.suptitle("MediaPipe Pose — Landmark Trajectories",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Plot → {path}")


def draw_skeleton(frame, landmarks, h, w):
    """
    Draw BlazePose-style skeleton: red lines + green dots with white border.
    Matches the original mp.solutions.pose visual exactly.
    """
    pts = {}
    # Build pixel positions, skip landmarks outside frame
    for idx, lm in enumerate(landmarks):
        px = int(lm.x * w)
        py = int(lm.y * h)
        pts[idx] = (px, py)

    # 1) Draw red bones FIRST (so dots appear on top)
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            vis_a = landmarks[a].visibility if landmarks[a].visibility is not None else 0
            vis_b = landmarks[b].visibility if landmarks[b].visibility is not None else 0
            if vis_a > 0.3 and vis_b > 0.3:   # only draw bone if both ends visible
                cv2.line(frame, pts[a], pts[b], BONE_COLOR, 2, cv2.LINE_AA)

    # 2) Draw green dots WITH white border on top of bones
    for idx, lm in enumerate(landmarks):
        vis = lm.visibility if lm.visibility is not None else 0
        if vis > 0.3:
            px, py = pts[idx]
            cv2.circle(frame, (px, py), DOT_RADIUS + DOT_BORDER,
                       (255, 255, 255), -1)           # white background
            cv2.circle(frame, (px, py), DOT_RADIUS,
                       JOINT_COLOR, -1, cv2.LINE_AA)  # green fill

    return pts


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    ensure_model()
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    ts_tag    = time.strftime("%Y%m%d_%H%M%S")
    csv_path  = os.path.join(OUTPUT_BASE, f"mp_log_{ts_tag}.csv")
    plot_path = os.path.join(OUTPUT_BASE, f"mp_plot_{ts_tag}.png")
    snap_dir  = os.path.join(OUTPUT_BASE, f"snapshots_{ts_tag}")

    # ── Open camera ──
    print(f"Opening camera {CAMERA_INDEX}…  (up to 12 s)")
    cap, backend = open_cam(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    if cap is None:
        print("ERROR: Could not open camera. Check CAMERA_INDEX at top of script.")
        return
    print(f"Camera OK  (backend={backend})\n")

    # ── Build PoseLandmarker (new Tasks API) ──
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,   # frame-by-frame with timestamps
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    landmark_names = list(TRACKED.keys())
    csv_rows         = []
    data_by_landmark = {n: [] for n in landmark_names}

    start_time = time.time()
    paused     = False
    snap_count = 0
    frame_ms   = 0          # timestamp in ms for VIDEO mode

    cv2.namedWindow("MediaPipe 3D Skeleton Motion Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MediaPipe 3D Skeleton Motion Tracking", CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2)

    print("="*55)
    print("  MediaPipe Pose — Laptop cam (index 0)")
    print(f"  Tracking: {', '.join(landmark_names)}")
    print("  Q → quit+save    P → pause/resume    S → snapshot")
    print("="*55 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            if paused:
                cv2.putText(frame, "PAUSED — press P to resume", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("MediaPipe 3D Skeleton Motion Tracking",
                           cv2.resize(frame, (CAMERA_WIDTH//2, CAMERA_HEIGHT//2)))
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"): break
                if key == ord("p"): paused = False
                continue

            h, w   = frame.shape[:2]
            elapsed = time.time() - start_time

            # ── Run landmarker ──
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_ms += int(1000 / CAMERA_FPS)   # simulate monotonic ms timestamp
            result    = landmarker.detect_for_video(mp_image, frame_ms)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]  # first (only) pose
                pts = draw_skeleton(frame, lms, h, w)

                # ── Extract & highlight tracked joints ──
                row = [f"{elapsed:.4f}"]
                for name in landmark_names:
                    idx = TRACKED[name]
                    lm  = lms[idx]
                    vis = lm.visibility if lm.visibility is not None else 0.0
                    if vis >= VISIBILITY_THRESH:
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        pz = lm.z
                        row += [px, py, f"{pz:.5f}", f"{vis:.3f}"]
                        # Tracked joint: cyan outer ring over the green dot
                        cv2.circle(frame, (px, py), DOT_RADIUS + 8,
                                   TRACK_COLOR, 2, cv2.LINE_AA)
                        cv2.putText(frame,
                                    name.replace("_"," "),
                                    (px+17, py+5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                    (255, 255, 255), 1)
                        data_by_landmark[name].append((px, py, pz, elapsed))
                    else:
                        row += [None, None, None, f"{vis:.3f}"]
                csv_rows.append(row)

            # ── HUD ──
            cv2.putText(frame,
                        "Logitech Cam 0 — Press 'q' to Quit  |  P=pause  S=snap",
                        (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 255), 2)
            cv2.putText(frame,
                        f"Logitech Tracking -> t={elapsed:.2f}s  frames={len(csv_rows)}",
                        (12, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (200, 200, 200), 1)

            cv2.imshow("MediaPipe 3D Skeleton Motion Tracking",
                       cv2.resize(frame, (CAMERA_WIDTH//2, CAMERA_HEIGHT//2)))

            key = cv2.waitKey(1) & 0xFF
            if   key == ord("q"): break
            elif key == ord("p"): paused = True
            elif key == ord("s"):
                os.makedirs(snap_dir, exist_ok=True)
                p = os.path.join(snap_dir, f"snap_{snap_count:04d}.jpg")
                cv2.imwrite(p, frame)
                snap_count += 1
                print(f"Snapshot → {p}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    # ── Save ──
    print("\nSaving outputs…")
    if csv_rows:
        save_csv(csv_path, csv_rows, landmark_names)
        save_plot(plot_path, data_by_landmark)
        print(f"\nDone!  {len(csv_rows)} frames logged → {OUTPUT_BASE}")
    else:
        print("No data recorded — make sure your body is visible to the camera.")


if __name__ == "__main__":
    main()
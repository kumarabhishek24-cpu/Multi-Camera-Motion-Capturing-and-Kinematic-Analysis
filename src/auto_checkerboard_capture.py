import cv2
import os
import time
import numpy as np

# ============================================
# CHECKERBOARD SETTINGS
# ============================================

# IMPORTANT:
# 8x6 squares  =>  7x5 inner corners

CHECKERBOARD = (5 , 7)

# ============================================
# SAVE PATH
# ============================================

save_path = "calibration_images/cam4"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# ============================================
# CAMERA INDEX
# ============================================

# 0 = laptop webcam
# 1 = first Logitech
# 2 = second Logitech

cap = cv2.VideoCapture(4, cv2.CAP_DSHOW)

# ============================================
# CAMERA CHECK
# ============================================

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

# ============================================
# CAPTURE SETTINGS
# ============================================

img_count = 0

capture_delay = 2  # seconds

last_capture_time = time.time()

print("===================================")
print("AUTO CHECKERBOARD CAPTURE STARTED")
print("Press 'q' to quit")
print("===================================")

# ============================================
# MAIN LOOP
# ============================================

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ============================================
    # IMAGE QUALITY ANALYSIS
    # ============================================

    # Blur Detection
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness Detection
    mean_brightness = np.mean(gray)

    # Edge Detection
    edges = cv2.Canny(gray, 50, 150)

    edge_count = np.sum(edges > 0)

    # ============================================
    # IMPROVED CHECKERBOARD DETECTION
    # ============================================

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    ret_corners, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags
    )

    # ============================================
    # SUCCESS CASE
    # ============================================

    if ret_corners:

        # Refine corners
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001
            )
        )

        print("SUCCESS: Checkerboard detected")

        # Draw corners
        cv2.drawChessboardCorners(
            frame,
            CHECKERBOARD,
            corners,
            ret_corners
        )

        # Display status
        cv2.putText(
            frame,
            "CHECKERBOARD DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Auto save after delay
        current_time = time.time()

        if current_time - last_capture_time > capture_delay:

            img_name = f"{save_path}/calib_{img_count}.png"

            cv2.imwrite(img_name, frame)

            print(f"SAVED: {img_name}")

            img_count += 1

            last_capture_time = current_time

    # ============================================
    # FAILURE CASE
    # ============================================

    else:

        print("FAILED: Checkerboard NOT detected")

        # Blur issue
        if laplacian_var < 100:

            reason = "BLUR DETECTED"

            print("REASON: Image blurry")

        # Dark image
        elif mean_brightness < 50:

            reason = "IMAGE TOO DARK"

            print("REASON: Poor lighting")

        # Bright image
        elif mean_brightness > 220:

            reason = "IMAGE TOO BRIGHT"

            print("REASON: Overexposed")

        # Low texture
        elif edge_count < 5000:

            reason = "LOW TEXTURE"

            print("REASON: Checkerboard not visible properly")

        else:

            reason = "CHECKERBOARD NOT FOUND"

            print("REASON: Wrong angle / partial visibility")

        # Display failure reason
        cv2.putText(
            frame,
            reason,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # ============================================
    # DISPLAY CAPTURE COUNT
    # ============================================

    cv2.putText(
        frame,
        f"Captured Images: {img_count}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # ============================================
    # SHOW FRAME
    # ============================================

    cv2.imshow("Auto Checkerboard Capture", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================
# CLEANUP
# ============================================

cap.release()

cv2.destroyAllWindows()

print("===================================")
print("CAPTURE SESSION ENDED")
print(f"TOTAL IMAGES CAPTURED: {img_count}")
print("===================================")
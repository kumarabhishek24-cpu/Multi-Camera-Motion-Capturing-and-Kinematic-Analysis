import cv2
import os
import time

# ==============================
# CHECKERBOARD SETTINGS
# ==============================

CHECKERBOARD = (7, 5)

# Camera indexes
cam1_index = 2
cam2_index = 1

# ==============================
# OPEN CAMERAS
# ==============================

cap1 = cv2.VideoCapture(cam1_index, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(cam2_index, cv2.CAP_DSHOW)

# ==============================
# CREATE FOLDERS
# ==============================

os.makedirs("stereo_images/cam1", exist_ok=True)
os.makedirs("stereo_images/cam2", exist_ok=True)

# ==============================
# START
# ==============================

img_count = 0
last_capture_time = time.time()

print("\n===================================")
print("AUTO STEREO CAPTURE STARTED")
print("===================================")
print("Show checkerboard to BOTH cameras")
print("Press Q to quit")
print("===================================\n")

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("ERROR: Cannot read cameras")
        break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect checkerboard
    ret_cb1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
    ret_cb2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)

    # Draw corners
    if ret_cb1:
        cv2.drawChessboardCorners(frame1, CHECKERBOARD, corners1, ret_cb1)

    if ret_cb2:
        cv2.drawChessboardCorners(frame2, CHECKERBOARD, corners2, ret_cb2)

    # Show windows
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    current_time = time.time()

    # BOTH cameras must detect checkerboard
    if ret_cb1 and ret_cb2:

        print("SUCCESS: Checkerboard detected in BOTH cameras")

        # Capture every 2 seconds
        if current_time - last_capture_time > 2:

            cam1_path = f"stereo_images/cam1/cam1_{img_count}.png"
            cam2_path = f"stereo_images/cam2/cam2_{img_count}.png"

            cv2.imwrite(cam1_path, frame1)
            cv2.imwrite(cam2_path, frame2)

            print(f"SAVED PAIR: {img_count}")

            img_count += 1
            last_capture_time = current_time

    else:

        if not ret_cb1 and not ret_cb2:
            print("FAILED: Checkerboard not detected in BOTH cameras")

        elif not ret_cb1:
            print("FAILED: Camera 1 cannot detect checkerboard")

        elif not ret_cb2:
            print("FAILED: Camera 2 cannot detect checkerboard")

    # Quit
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap1.release()
cap2.release()

cv2.destroyAllWindows()

print("\n===================================")
print(f"TOTAL STEREO PAIRS CAPTURED: {img_count}")
print("===================================")
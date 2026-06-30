import cv2
import numpy as np

# ==========================================
# LOAD CALIBRATION
# ==========================================
data = np.load("calibration/logitech_cam1_calibration.npz")

camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

# ==========================================
# OPEN CAMERA
# ==========================================
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

print("Press Q to quit")

# ==========================================
# LOOP
# ==========================================
while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]

    # ==========================================
    # UNDISTORT
    # ==========================================
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        1,
        (w, h)
    )

    undistorted = cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

    # ==========================================
    # SHOW
    # ==========================================
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()
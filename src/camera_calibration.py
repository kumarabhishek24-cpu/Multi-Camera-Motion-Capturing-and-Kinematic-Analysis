import cv2
import numpy as np
import glob
import os

# ==========================================
# CHECKERBOARD SETTINGS
# ==========================================
CHECKERBOARD = (7, 5)

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

# ==========================================
# PREPARE OBJECT POINTS
# ==========================================
objp = np.zeros(
    (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# ==========================================
# LOAD IMAGES
# ==========================================
images = glob.glob("calibration_images/cam4/*.png")

print(f"\nFound {len(images)} calibration images\n")

valid_images = 0

# ==========================================
# PROCESS IMAGES
# ==========================================
for fname in images:

    img = cv2.imread(fname)

    if img is None:
        print(f"Could not read: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # NEW SB DETECTOR
    # ==========================================
    ret, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_EXHAUSTIVE
    )

    if ret:

        valid_images += 1

        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(
            img,
            CHECKERBOARD,
            corners,
            ret
        )

        print(f"SUCCESS: {fname}")

        cv2.imshow("Detected", img)
        cv2.waitKey(100)

    else:
        print(f"FAILED: {fname}")

cv2.destroyAllWindows()

# ==========================================
# CHECK VALID IMAGES
# ==========================================
print("\n==============================")
print(f"VALID IMAGES: {valid_images}")
print("==============================")

if valid_images < 10:
    print("\nERROR: Not enough valid images")
    exit()

# ==========================================
# CAMERA CALIBRATION
# ==========================================
print("\nRunning Camera Calibration...\n")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

# ==========================================
# REPROJECTION ERROR
# ==========================================
mean_error = 0

for i in range(len(objpoints)):

    imgpoints2, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(
        imgpoints[i],
        imgpoints2,
        cv2.NORM_L2
    ) / len(imgpoints2)

    mean_error += error

mean_error /= len(objpoints)

# ==========================================
# SAVE CALIBRATION
# ==========================================
os.makedirs("calibration", exist_ok=True)

np.savez(
    "calibration/cam4_calibration.npz",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs
)

# ==========================================
# RESULTS
# ==========================================
print("\n===================================")
print("CALIBRATION SUCCESSFUL")
print("===================================\n")

print("Camera Matrix:\n")
print(camera_matrix)

print("\nDistortion Coefficients:\n")
print(dist_coeffs)

print(f"\nMean Reprojection Error: {mean_error}")

print("\nSaved File:")
print("calibration/logitech_cam1_calibration.npz")
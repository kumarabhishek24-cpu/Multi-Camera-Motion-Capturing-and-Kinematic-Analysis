import cv2
import time

# Open webcam
cap = cv2.VideoCapture(0)

# Image counter
img_counter = 0

# Time interval between captures (in seconds)
capture_interval = 1

# Store last capture time
last_capture_time = time.time()

# Check camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Show live feed
    cv2.imshow("Automatic Frame Capture", frame)

    # Current time
    current_time = time.time()

    # Automatically save frame every interval
    if current_time - last_capture_time >= capture_interval:

        img_name = f"output/frame_{img_counter}.png"

        cv2.imwrite(img_name, frame)

        print(f"Saved: {img_name}")

        img_counter += 1

        last_capture_time = current_time

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
import cv2

# Open webcam
cap = cv2.VideoCapture(0)

# Check webcam
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # LED counter
    led_id = 0

    # Process contours
    for contour in contours:

        # Ignore tiny noise
        area = cv2.contourArea(contour)

        if area > 50:

            # Calculate moments
            M = cv2.moments(contour)

            if M["m00"] != 0:

                # Compute centroid
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw centroid
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                # LED label
                label = f"LED {led_id}"

                # Display label
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2
                )

                # Display coordinates
                cv2.putText(
                    frame,
                    f"({cx}, {cy})",
                    (cx + 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

                # Print coordinates
                print(f"{label}: X={cx}, Y={cy}")

                led_id += 1

    # Show frames
    cv2.imshow("Multi-LED Tracking", frame)
    cv2.imshow("Threshold", threshold)

    # Quit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
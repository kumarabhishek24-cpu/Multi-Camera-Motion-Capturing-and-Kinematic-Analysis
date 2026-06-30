import cv2
import argparse


def main():
    parser = argparse.ArgumentParser(description="Find working camera indexes")
    parser.add_argument("--max-index", type=int, default=10, help="Scan camera indexes from 0 to max-index-1")
    parser.add_argument(
        "--preview-ms",
        type=int,
        default=1200,
        help="Preview duration in milliseconds when a camera is found",
    )
    parser.add_argument(
        "--wait-key",
        action="store_true",
        help="Wait for a key press on each found camera preview (old behavior)",
    )
    args = parser.parse_args()

    print("Checking camera indexes...\n")
    print("Tip: press Q in a preview window to stop early.\n")

    found = []

    for i in range(args.max_index):
        print(f"Testing camera index: {i}")

        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                found.append(i)
                print(f"Camera {i} working")

                window_name = f"Camera Index {i}"
                cv2.imshow(window_name, frame)

                if args.wait_key:
                    print("Press any key for next camera...")
                    key = cv2.waitKey(0) & 0xFF
                else:
                    print(f"Showing preview for {args.preview_ms} ms...")
                    key = cv2.waitKey(max(1, args.preview_ms)) & 0xFF

                cv2.destroyWindow(window_name)

                if key == ord("q"):
                    print("Stopped by user.")
                    cap.release()
                    break

        cap.release()

    if found:
        print(f"\nFound working camera indexes: {found}")
    else:
        print("\nNo working cameras found in scanned range.")

    print("Done")


if __name__ == "__main__":
    main()
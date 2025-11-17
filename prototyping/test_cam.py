import argparse
import cv2
import time

def main(fps):
    print(cv2.getBuildInformation())
    pipeline = (
        "v4l2src ! "
        "image/jpeg,width=1280,height=800,framerate=100/1 ! "
        "jpegdec ! videoconvert ! appsink"
    )

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    prev_time = time.time()
    cur_time = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        cur_time = time.time()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame
        if fps:
            print(f"{1 / (cur_time - prev_time)} FPS")
        else:
            cv2.putText(frame, f"{1 / (cur_time - prev_time)} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.imshow("Webcam Feed", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_time = cur_time

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Flags for the type of test on the camera')
parser.add_argument('-f', '--fps', action='store_true', help='Run this script in FPS mode')
args = parser.parse_args()
if __name__ == "__main__":
    main(args.fps)

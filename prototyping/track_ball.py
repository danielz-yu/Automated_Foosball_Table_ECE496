import cv2
from math import sin, cos, atan
from multiprocessing import Pool, Queue
import numpy as np
import time

lower_orange = np.array([9 / 2, 255 * .50, 0])
upper_orange = np.array([19 / 2, 255 * .84, 255])

def init_pool(d_b):
    global detection_buffer
    detection_buffer = d_b


def detect_object(frame):
    detection_buffer.put(frame)


def show():
    global last_five, total_x, total_y, avg_x, avg_y, id, positive, out
    while True:
        frame = detection_buffer.get()
        if frame is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            max_area = 0
            x, y, w, h = 0, 0, 0, 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    x, y, w, h = cv2.boundingRect(cnt)
            pos = (x + w // 2, y + h // 2)
            total_x -= last_five[id][0]
            total_x += pos[0]
            total_y -= last_five[id][1]
            total_y += pos[1]
            avg_x = total_x // 5
            avg_y = total_y // 5
            last_five[id] = pos
            multiply_xy = 0
            square_x = 0
            for p in last_five:
                multiply_xy += (p[0] - avg_x) * (p[1] - avg_y)
                square_x += (p[0] - avg_x) ** 2
            if square_x != 0:
                angle = atan(multiply_xy / square_x)
                if pos[0] > avg_x:
                    cv2.line(frame, pos, (pos[0] + int(100 * cos(angle)), pos[1] + int(100 * sin(angle))), (0, 0, 255), 2)
                else:
                    cv2.line(frame, pos, (pos[0] - int(100 * cos(angle)), pos[1] - int(100 * sin(angle))), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            id += 1
            id %= 5
            cv2.imshow("Video", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

# takes in input MP4 video and draws boundary of ball through color masking and line of direction calculated based on last 5 frame positions
last_five = [(0, 0)] * 5
total_x = 0
total_y = 0
avg_x = 0
avg_y = 0
id = 0
positive = True

# required for Windows:
if __name__ == "__main__":
    start = time.time()
    # right now this is set up to use the Arducam as input instead of MP4 vids, but CV2 documentation has instruction on how to run with files
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    detection_buffer = Queue()
    pool = Pool(5, initializer=init_pool, initargs=(detection_buffer,))
    # run the "show" task:
    show_future = pool.apply_async(show)
    futures = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f = pool.apply_async(detect_object, args=(frame,))
        futures.append(f)
        time.sleep(0.001)
    # wait for all the frame-putting tasks to complete:
    for f in futures:
        f.get()

    # signal the "show" task to end by placing None in the queue
    detection_buffer.put(None)
    show_future.get()
    end = time.time()
    print(f"Done in {end - start} seconds.")

    cap.release()
    cv2.destroyAllWindows()
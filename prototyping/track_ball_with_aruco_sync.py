import collections
import cv2
import numpy as np
import time
from math import sin, cos, atan
from multiprocessing import Pool, Queue

# --------------------
# Ball color thresholds
# --------------------
lower_orange = np.array([9 / 2, 255 * .50, 0])
upper_orange = np.array([19 / 2, 255 * .84, 255])

# --------------------
# ArUco setup
# --------------------
aruco_dict_type = cv2.aruco.DICT_4X4_50
corner_ids = [0, 1, 2, 3]   # top-left, top-right, bottom-right, bottom-left
W, H = 800, 600             # warped output size

aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# --------------------
# Multiprocessing setup
# --------------------
def init_pool(d_b):
    global detection_buffer
    detection_buffer = d_b

def detect_object(frame):
    detection_buffer.put(frame)

history = collections.deque(maxlen=5)

def show():
    while True:
        frame = detection_buffer.get()
        if frame is None:
            break

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
        if max_area > 0:
            M = cv2.moments(contours[np.argmax([cv2.contourArea(c) for c in contours])])
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
                pos = (cx, cy)

                history.append(pos)

                # Only draw when we have at least 2 points
                if len(history) >= 2:
                    # --- velocity estimate over time ---
                    # t = 0..n-1 (uniform dt), least-squares slope for x(t), y(t)
                    n = len(history)
                    t = np.arange(n, dtype=np.float32)
                    tx = t - t.mean()
                    # convert to array
                    pts = np.array(history, dtype=np.float32)
                    xs = pts[:,0]; ys = pts[:,1]
                    denom = (tx**2).sum()
                    if denom > 0:
                        dx_dt = (tx * xs).sum() / denom
                        dy_dt = (tx * ys).sum() / denom
                    else:
                        dx_dt = xs[-1] - xs[0]
                        dy_dt = ys[-1] - ys[0]

                    # Fallback if tiny motion
                    if abs(dx_dt) + abs(dy_dt) < 1e-3:
                        dx_dt = xs[-1] - xs[-2]
                        dy_dt = ys[-1] - ys[-2]

                    angle = np.arctan2(dy_dt, dx_dt)
                    # speed magnitude
                    speed = np.hypot(dx_dt, dy_dt)

                    # map speed to arrow length
                    # adjust "scale" to taste (pixels per unit speed)
                    scale = 5.0
                    length = int(scale * speed)

                    # clamp length so it never gets too short or too long
                    length = max(20, min(length, 200))

                    # compute endpoint
                    end = (int(pos[0] + length * np.cos(angle)),
                        int(pos[1] + length * np.sin(angle)))
                    cv2.arrowedLine(frame, pos, end, (0,0,255), 2, tipLength=0.25)

                    # Debug overlay
                    speed = np.hypot(dx_dt, dy_dt)
                    cv2.putText(frame, f"angle:{np.degrees(angle):.1f} deg  speed:{speed:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Warped + Ball Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    start = time.time()
    cap = cv2.VideoCapture(0)  # or webcam index

    # --- Detect ArUco in first frame ---
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        exit()

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None or len(ids) < 4:
        print("Not all 4 ArUco markers detected.")
        exit()

    # Find centers of detected markers
    id_to_center = {}
    for id, c in zip(ids, corners):
        pts = c[0]
        center = np.mean(pts, axis=0)
        id_to_center[id[0]] = center

    src_pts = []
    for target_id in corner_ids:
        if target_id not in id_to_center:
            print(f"Marker ID {target_id} not found.")
            exit()
        src_pts.append(id_to_center[target_id])
    src_pts = np.array(src_pts, dtype="float32")

    dst_pts = np.array([
        [0, 0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0, H - 1]
    ], dtype="float32")

    H_matrix, _ = cv2.findHomography(src_pts, dst_pts)

    # --------------------
    # Multiprocessing loop
    # --------------------
    detection_buffer = Queue()
    pool = Pool(5, initializer=init_pool, initargs=(detection_buffer,))
    show_future = pool.apply_async(show)
    futures = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to beginning
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Warp frame before ball detection
        warped = cv2.warpPerspective(frame, H_matrix, (W, H))

        f = pool.apply_async(detect_object, args=(warped,))
        futures.append(f)
        time.sleep(0.001)

    for f in futures:
        f.get()

    detection_buffer.put(None)
    show_future.get()

    end = time.time()
    print(f"Done in {end - start:.2f} seconds.")

    cap.release()
    cv2.destroyAllWindows()

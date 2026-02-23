import collections
import math
import cv2
import numpy as np
import time
from math import sin, cos, atan
from multiprocessing import Pool, Queue

from arucos import process_aruco, get_perspective_matrix

# --------------------
# Ball color thresholds
# --------------------
lower_orange = np.array([9 / 2, 255 * .50, 0])
upper_orange = np.array([19 / 2, 255 * .84, 255])

colors_low = [np.array([0, 255 * .3, 70]), np.array([350 / 2, 255 * .3, 70])]
colors_high = [np.array([10 / 2, 255 * .95, 255]), np.array([180, 255 * .95, 255])]

location_start = [0, 100, 300, 500]
location_end = [25, 125, 325, 525]
num_foosmen = [1, 2, 5, 3]

# --------------------
# ArUco setup
# --------------------
NUM_ARUCOS = 12
rod_ids = [(4, 5), (6, 7), (8, 9), (10, 11)] # from the automatic rod closest to own goal to the farthest
rod_centers = {}
W, H = 800, 600             # warped output size


KICK_THRESHOLD = 7
LENGTH_THRESHOLD = 0

# --------------------
# Multiprocessing setup
# --------------------
def init_pool(d_b):
    global detection_buffer
    detection_buffer = d_b

def detect_object(frame):
    detection_buffer.put(frame)

history = collections.deque(maxlen=5)

import numpy as np

def clip_line_to_frame(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    points = []

    # Avoid division by zero
    if abs(dx) > 1e-6:
        # left edge (x=0)
        t = -x1 / dx
        y = y1 + t*dy
        if 0 <= y <= H-1:
            points.append((0, int(round(y))))
        # right edge (x=W-1)
        t = (W-1 - x1) / dx
        y = y1 + t*dy
        if 0 <= y <= H-1:
            points.append((W-1, int(round(y))))

    if abs(dy) > 1e-6:
        # top edge (y=0)
        t = -y1 / dy
        x = x1 + t*dx
        if 0 <= x <= W-1:
            points.append((int(round(x)), 0))
        # bottom edge (y=H-1)
        t = (H-1 - y1) / dy
        x = x1 + t*dx
        if 0 <= x <= W-1:
            points.append((int(round(x)), H-1))

    # We expect 2 valid intersections
    if len(points) >= 2:
        return points[0], points[1]
    else:
        return None, None

def percent_along_line(rod_pt1, rod_pt2, intersection):
    p1 = np.array(rod_pt1, dtype=float)
    p2 = np.array(rod_pt2, dtype=float)
    pi = np.array(intersection, dtype=float)

    rod_vec = p2 - p1
    inter_vec = pi - p1

    rod_len = np.linalg.norm(rod_vec)
    if rod_len < 1e-6:
        return 0.0  # degenerate case
    
    # projection of inter_vec onto rod_vec
    t = np.dot(inter_vec, rod_vec) / (rod_len**2)

    # percentage along the rod
    return t * 100.0

def point_on_line(rod_pt1, rod_pt2, percent):
    """
    rod_pt1s[i], rod_pt2s[i]: (x,y) endpoints of rod line (clipped to frame)
    percent: float in [0,100], percentage along rod_pt1→rod_pt2
    """
    p1 = np.array(rod_pt1, dtype=float)
    p2 = np.array(rod_pt2, dtype=float)
    t = percent / 100.0
    return tuple((p1 + t*(p2 - p1)).astype(int))


def show(rod_slopes, rod_intercepts, rod_pt1s, rod_pt2s):
    while True:
        frame = detection_buffer.get()[10:-20, 30:-20]
        if frame is None:
            break
        frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=10)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        # v = np.clip(v * 0.7, 0, 255).astype(np.uint8)  # reduce brightness
        # hsv = cv2.merge((h, s, v))
        mask = cv2.inRange(hsv, colors_low[0], colors_high[0])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        mask1 = cv2.inRange(hsv, colors_low[1], colors_high[1])
        mask1 = cv2.erode(mask1, None, iterations=2)
        mask1 = cv2.dilate(mask1, None, iterations=2)
        
        mask = cv2.bitwise_or(mask, mask1)

        for i in range(len(num_foosmen)):
            contours, _ = cv2.findContours(mask[:, location_start[i]:location_end[i]], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 2. Sort contours by area (descending)
            # We use cv2.contourArea as the key for sorting
            cnts_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

            # 3. Select the top N contours, where N is num_foosmen[i]
            # Python list slicing [:N] handles cases where len(cnts) < N automatically
            target_contours = cnts_sorted[:num_foosmen[i]]

            # 4. Loop through the selected top contours and draw
            for cnt in target_contours:
                x1, y1, w1, h1 = cv2.boundingRect(cnt)
                cv2.rectangle(frame, 
                      (x1 + location_start[i], y1), 
                      (x1 + location_start[i] + w1, y1 + h1), 
                      (0, 255, 0), 2)
                #x1, y1, w1, h1 = 0, 0, 0, 0
            #max_area = 0
            #for cnt in contours:
                #area = cv2.contourArea(cnt)
               # if area > max_area:
              #      max_area = area
             #       x1, y1, w1, h1 = cv2.boundingRect(cnt)
            #cv2.rectangle(frame, (x1 + location_start[i], y1), (x1 + location_start[i] + w1, y1 + h1), (0, 255, 0), 2)
        # for i in range(len(rod_slopes)):
            # cv2.line(frame, (0, int(rod_intercepts[i])), (W, int(rod_intercepts[i] + rod_slopes[i] * W)), (0, 0, 255), 2)
        cv2.imshow("Warped + Ball Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

def init(fps):
    # Can confirm using this printout to see whether GStreamer support is enabled on the version of OpenCV used
    print(cv2.getBuildInformation())

    # Define pipeline based on OV9782 specs
    pipeline = (
        "v4l2src ! "
        f"image/jpeg,width=1280,height=800,framerate={fps}/1 ! "
        "jpegdec ! videoconvert ! appsink"
    )

    # Use pipeline to define video capture object through GStreamer
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap


def get_rod_lines(id_to_center, H_matrix):
    rod_slopes, rod_intercepts = [], []
    rod_pt1s, rod_pt2s = [], []
    
    for rod in rod_ids:
        # Convert points of both ends of the rod to warped coords
        pts = np.array([id_to_center[rod[0]], id_to_center[rod[1]]], dtype="float32").reshape(-1,1,2)
        pts_warped = cv2.perspectiveTransform(pts, H_matrix)

        (x1,y1), (x2,y2) = pts_warped[:,0,:]

        # Append slope, intercept, point 1 locations, point 2 locations to respective vector
        rod_slopes.append((y2 - y1) / (x2 - x1))
        rod_intercepts.append(y1 - rod_slopes[-1] * x1)
        rod_pt1, rod_pt2 = clip_line_to_frame(x1, y1, x2, y2)
        rod_pt1s.append(rod_pt1)
        rod_pt2s.append(rod_pt2)
    # Debug output
    print(f"Slopes: {rod_slopes}\nIntercepts:{rod_intercepts}\nPoint 1s:{rod_pt1s}\nPoint 2s:{rod_pt2s}")
    return rod_slopes, rod_intercepts, rod_pt1s, rod_pt2s


if __name__ == "__main__":
    # Set up camera backend and retrieve video capture object
    cap = init(100)

    id_to_center = process_aruco(cap, NUM_ARUCOS)
    
    cap.release()
    cap = init(30)

    H_matrix = get_perspective_matrix(id_to_center, W, H)

    rod_slopes, rod_intercepts, rod_pt1s, rod_pt2s = get_rod_lines(id_to_center, H_matrix)
    for i in range(len(rod_intercepts)):
        rod_intercepts[i] += rod_slopes[i] * 20
    # --------------------
    # Multiprocessing loop
    # --------------------
    detection_buffer = Queue()
    pool = Pool(5, initializer=init_pool, initargs=(detection_buffer,))
    show_future = pool.apply_async(show, args=(rod_slopes, rod_intercepts, rod_pt1s, rod_pt2s))
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

    cap.release()
    cv2.destroyAllWindows()

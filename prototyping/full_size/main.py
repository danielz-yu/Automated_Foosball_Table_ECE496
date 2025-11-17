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

# lower_pink = np.array([340 / 2, 255 * .05, 0])
# upper_pink = np.array([380 / 2, 255 * .4, 255])
partition_1 = 35.71
partition_2 = 64.29
gap = 28.57

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
    rod_pt1, rod_pt2: (x,y) endpoints of rod line (clipped to frame)
    percent: float in [0,100], percentage along rod_pt1→rod_pt2
    """
    p1 = np.array(rod_pt1, dtype=float)
    p2 = np.array(rod_pt2, dtype=float)
    t = percent / 100.0
    return tuple((p1 + t*(p2 - p1)).astype(int))


def reflect_trajectory(pos, vel, max_bounces=5, line_slope=None, line_intercept=None):
    """
    Simulate a straight-line ball trajectory with specular reflections off the
    rectangle [0,W] x [0,H], and optionally stop when it hits a target line.

    pos: (x, y) starting position (pixels, top-left origin)
    vel: (vx, vy) velocity vector (pixels per step; direction & relative speed)
    W,H: frame/table dimensions (in pixels)
    max_bounces: maximum number of wall bounces to simulate
    line_slope, line_intercept: y = m*x + b (if target line is not vertical)
    vertical_x: x = c for a vertical target line (if used)
    Returns:
      - (intersection_point or None, list_of_polyline_points_along_path)
    """

    # Unpack starting point and velocity
    x, y = pos
    vx, vy = vel

    # We’ll record the vertices of each segment in this list (for drawing)
    trajectory_points = [(x, y)]

    # Simulate up to max_bounces reflections
    for _ in range(max_bounces):

        # --- Time (parametric t) to each wall from current (x,y) with velocity (vx,vy)
        # Initialize as "no hit" (infinite time) by default.
        tx = float('inf')
        ty = float('inf')

        # If moving right, time to right wall is distance / speed; if left, time to left wall.
        if abs(vx) < 1e-6 or abs(vy) < 1e-6:
            break
        if vx > 0:
            tx = (W - x) / vx
        elif vx < 0:
            tx = -x / vx   # vx < 0 makes this positive

        # Same logic vertically: down to bottom wall, up to top wall.
        if vy > 0:
            ty = (H - y) / vy
        elif vy < 0:
            ty = -y / vy   # vy < 0 makes this positive

        # The first wall we’ll hit is the one with the smaller positive time.
        tmin = min(tx, ty)
        if tmin == float('inf'):
            # Not moving or pointing outward with no wall intersection: stop.
            break

        # Advance to the first collision point
        x_new = x + vx * tmin
        y_new = y + vy * tmin

        # --- Before we reflect, check if this segment hits the target line ---

        if line_slope is not None:
            # Solve for t where parametric segment hits y = m*x + b:
            # y + vy*t = m*(x + vx*t) + b
            # t * (vy - m*vx) = m*x + b - y
            A = vy - line_slope * vx
            B = line_slope * x + line_intercept - y
            if abs(A) > 1e-6:
                t_line = B / A
                # If that t falls within this segment (0..tmin), we intersect before bouncing.
                if 0 <= t_line <= tmin:
                    xi = x + vx * t_line
                    yi = y + vy * t_line
                    return (xi, yi), trajectory_points + [(xi, yi)]

        # No intersection this leg: accept the bounce point
        x, y = x_new, y_new
        trajectory_points.append((x, y))

        # Reflect velocity depending on which wall we hit first:
        if tx < ty:
            # Hit a vertical wall (left or right): flip horizontal component
            vx = -vx
        else:
            # Hit a horizontal wall (top or bottom): flip vertical component
            vy = -vy

    # No intersection found within the allowed bounces; return the path we traced.
    return None, trajectory_points


def show(rod_slopes, rod_intercepts, rod_pt1s, rod_pt2s):
    while True:
        frame = detection_buffer.get()
        if frame is None:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        # v = np.clip(v * 0.7, 0, 255).astype(np.uint8)  # reduce brightness
        # hsv = cv2.merge((h, s, v))
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        x, y, w, h = 0, 0, 0, 0
        dx_dt, dy_dt = 0, 0
        speed = 0
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

        for i in range(len(rod_slopes)):
            rod_slope = rod_slopes[i]
            rod_intercept = rod_intercepts[i]
            rod_pt1 = rod_pt1s[i]
            rod_pt2 = rod_pt2s[i]
            # If ball is behind the rod in question, ignore. There is no need to intercept.
            if not rod_pt1 or x < min(rod_pt1[0], rod_pt2[0]):
                continue
            cv2.line(frame, (0, int(rod_intercept)), (W, int(rod_intercept + rod_slope * W)), (0, 0, 255), 2)

            intersection, traj_pts = reflect_trajectory(
                pos, (dx_dt, dy_dt),
                max_bounces=5,
                line_slope=rod_slope,
                line_intercept=rod_intercept
            )

            # Draw the trajectory polyline
            # lengths_sum = 0
            for i in range(len(traj_pts)-1):
                cv2.line(frame, (int(traj_pts[i][0]), int(traj_pts[i][1])),
                            (int(traj_pts[i+1][0]), int(traj_pts[i+1][1])),
                            (255,0,0), 1)
            if len(traj_pts) == 2:
                lengths_sum = math.sqrt((int(traj_pts[0][0]) - int(traj_pts[1][0])) ** 2 + (int(traj_pts[0][1]) - int(traj_pts[1][1])) ** 2)
                if speed:
                    timed = lengths_sum / speed
            
                    # Detect when to kick and simulate; this will be replaced by a kick signal in the future (but only once for the first trigger into threshold)
                    if (timed < 0 and timed > -KICK_THRESHOLD) or (timed > 0 and timed < KICK_THRESHOLD):
                        cv2.line(frame, (0, int(rod_intercept)), (W, int(rod_intercept + rod_slope * W)), (0, 255, 255), 5)

            # Draw intersection if found
            if intersection:
                cv2.circle(frame, (int(intersection[0]), int(intersection[1])),
                        6, (0,255,255), -1)
                percentage = percent_along_line(rod_pt1, rod_pt2, intersection)
                # For reference purposes, we will draw the point of the three players which will be at the desired point
                pt1, pt2, pt3 = (0,0), (0,0), (0,0)
                if percentage < partition_1:
                    pt1 = point_on_line(rod_pt1, rod_pt2, percentage)
                    pt2 = point_on_line(rod_pt1, rod_pt2, percentage + gap)
                    pt3 = point_on_line(rod_pt1, rod_pt2, percentage + 2 * gap)
                elif percentage < partition_2:
                    pt1 = point_on_line(rod_pt1, rod_pt2, percentage - gap)
                    pt2 = point_on_line(rod_pt1, rod_pt2, percentage)
                    pt3 = point_on_line(rod_pt1, rod_pt2, percentage + gap)
                else:
                    pt1 = point_on_line(rod_pt1, rod_pt2, percentage - 2 * gap)
                    pt2 = point_on_line(rod_pt1, rod_pt2, percentage - gap)
                    pt3 = point_on_line(rod_pt1, rod_pt2, percentage)
                cv2.circle(frame, pt1, 5, (255, 255, 255), -1)
                cv2.circle(frame, pt2, 5, (255, 255, 255), -1)
                cv2.circle(frame, pt3, 5, (255, 255, 255), -1)
        cv2.imshow("Warped + Ball Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return

def init():
    # Can confirm using this printout to see whether GStreamer support is enabled on the version of OpenCV used
    print(cv2.getBuildInformation())

    # Define pipeline based on OV9782 specs
    pipeline = (
        "v4l2src ! "
        "image/jpeg,width=1280,height=800,framerate=100/1 ! "
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
    cap = init()

    id_to_center = process_aruco(cap, NUM_ARUCOS)

    H_matrix = get_perspective_matrix(id_to_center, W, H)

    rod_slopes, rod_intercepts, rod_pt1s, rod_pt2s = get_rod_lines(id_to_center, H_matrix)
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

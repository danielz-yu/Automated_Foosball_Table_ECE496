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

lower_pink = np.array([340 / 2, 255 * .05, 0])
upper_pink = np.array([380 / 2, 255 * .4, 255])

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

        # elif vertical_x is not None:
        #     # Target is the vertical line x = c. Check if the segment crosses x=c.
        #     # Crossing test in x: endpoints on opposite sides (or touching)
        #     if (x - vertical_x) * (x_new - vertical_x) <= 0:
        #         if vx != 0:
        #             t_line = (vertical_x - x) / vx
        #             # Confirm it's within the current segment
        #             if 0 <= t_line <= tmin:
        #                 yi = y + vy * t_line
        #                 return (vertical_x, yi), trajectory_points + [(vertical_x, yi)]

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


def show():
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

        mask = cv2.inRange(hsv, lower_pink, upper_pink)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area_1 = 0
        max_area_2 = 0
        x, y, w, h = 0, 0, 0, 0
        x2, y2, w2, h2 = 0, 0, 0, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area_1:
                max_area_2 = max_area_1
                max_area_1 = area
                x2, y2, w2, h2 = x, y, w, h
                x, y, w, h = cv2.boundingRect(cnt)
            elif area > max_area_2:
                max_area_2 = area
                x2, y2, w2, h2 = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
        x_line, y_line, x_line_2, y_line_2 = x + w // 2, y + h // 2, x2 + w2 // 2, y2 + h2 // 2
        if x_line_2 != x_line:
            slope = (y_line_2 - y_line) / (x_line_2 - x_line)
            intercept = int(y_line - slope * x_line)
            cv2.line(frame, (0, intercept), (W, int(intercept + slope * W)), (0, 0, 255), 2)
        
            if speed > 2.0:
                intersection, traj_pts = reflect_trajectory(
                    pos, (dx_dt, dy_dt),
                    max_bounces=5,
                    line_slope=slope,
                    line_intercept=intercept
                )

                # Draw the trajectory polyline
                for i in range(len(traj_pts)-1):
                    cv2.line(frame, (int(traj_pts[i][0]), int(traj_pts[i][1])),
                                (int(traj_pts[i+1][0]), int(traj_pts[i+1][1])),
                                (255,0,0), 1)

                # Draw intersection if found
                if intersection:
                    cv2.circle(frame, (int(intersection[0]), int(intersection[1])),
                            6, (0,255,255), -1)
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

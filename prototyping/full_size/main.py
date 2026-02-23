import collections
import math
import cv2
import numpy as np
import signal
import time
from math import sin, cos, atan
from multiprocessing import Pool, Queue, Array
import threading

from arucos import process_aruco, get_perspective_matrix
from motor_control import init_control, kick

availability = Array('b', 4)
for i in range(4):
	availability[i] = True

def receive(ser):
	global availability
	while True:
		try:
			if ser.in_waiting > 0:
				# Read the line until a \n is found
				response = ser.readline()

				# Decode bytes back into a string and strip whitespace
				decoded_response = response.decode('utf-8').strip()
				if decoded_response.startswith("MOVING GOALIE"):
					availability[1] = False
					decoded_response += f' {time.time()}'
				else:
					if decoded_response == 'ad':
						availability[0] = True
					elif decoded_response == 'bd':
						availability[1] = True
				print(f"Received: {decoded_response}")
		except serial.SerialException as e:
			print(f"Error: Could not open serial port {SERIAL_PORT}. Is it plugged in?")
			print(f"Details: {e}")
		except Exception as e:
			print(f"An unexpected error occurred: {e}")

# --------------------
# Ball color thresholds
# --------------------
lower_orange = np.array([15 / 2, 255 * .35, 10])
upper_orange = np.array([45 / 2, 255 * .65, 255])

# lower_pink = np.array([340 / 2, 255 * .05, 0])
# upper_pink = np.array([380 / 2, 255 * .4, 255])
# partition_1 = 35.71
# partition_2 = 64.29
# gap = 28.57

colors_low = [np.array([0, 255 * .3, 70]), np.array([350 / 2, 255 * .3, 70])]
colors_high = [np.array([10 / 2, 255 * .95, 255]), np.array([180, 255 * .95, 255])]

# location_start = [0, 100, 300, 500]
# location_end = [25, 125, 325, 525]
# num_foosmen = [1, 2, 5, 3]
# avg_pos = [-100, -100, -100, -100]
location_start = [0, 100]
location_end = [25, 125]
num_foosmen = [1, 2]
avg_pos = [-100, -100]

# First rod: 1 foosman, limited in movement unlike other rods 68 21
# min_first = 30.88
# max_first = 69.12
min_first = 40
max_first = 58

# Second rod: 2 foosmen
# Third rod: 5 foosmen
# Fourth rod: 3 foosmen
# partitions = [(100.0), (50.0, 100.0), (20.0, 40.0, 60.0, 80.0, 100.0), (33.333, 66.667, 100.0)]
# gaps = [100.0, 50.0, 20.0, 33.333]
partitions = [(100.0), (50.0, 100.0)]
gaps = [100.0, 50.0]

# --------------------
# ArUco setup
# --------------------
NUM_ARUCOS = 12
rod_ids = [(4, 5), (6, 7), (8, 9), (10, 11)] # from the automatic rod closest to own goal to the farthest
rod_centers = {}
W, H = 800, 600             # warped output size


KICK_THRESHOLD = 5
LENGTH_THRESHOLD = 25

# --------------------
# Multiprocessing setup
# --------------------
def init_pool(d_b):
    global detection_buffer
    detection_buffer = d_b

def detect_object(frame):
    detection_buffer.put(frame)

history = collections.deque(maxlen=5)

# --------------------
# Serial connection
# --------------------
ser = None

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
            points.append((0 - 20, int(round(y))))
        # right edge (x=W-1)
        t = (W-1 - x1) / dx
        y = y1 + t*dy
        if 0 <= y <= H-1:
            points.append((W-1 - 20, int(round(y))))

    if abs(dy) > 1e-6:
        # top edge (y=0)
        t = -y1 / dy
        x = x1 + t*dx
        if 0 <= x <= W-1:
            points.append((int(round(x)) - 20, 0))
        # bottom edge (y=H-1)
        t = (H-1 - y1) / dy
        x = x1 + t*dx
        if 0 <= x <= W-1:
            points.append((int(round(x)) - 20, H-1))

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


def reflect_trajectory(pos, vel, max_bounces=2, line_slope=None, line_intercept=None):
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

        if line_slope is not None and vx < 0:
            for i in range(len(line_slope) - 2, -1, -1):
                # Solve for t where parametric segment hits y = m*x + b:
                # y + vy*t = m*(x + vx*t) + b
                # t * (vy - m*vx) = m*x + b - y
                A = vy - line_slope[i] * vx
                B = line_slope[i] * x + line_intercept[i] - y
                if abs(A) > 1e-6:
                    t_line = B / A
                    # If that t falls within this segment (0..tmin), we intersect before bouncing.
                    if 0 <= t_line <= tmin:
                        xi = x + vx * t_line
                        yi = y + vy * t_line
                        return (xi, yi, i), trajectory_points + [(xi, yi)]

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


def show(rod_slopes, rod_intercepts, rod_pt1s, rod_pt2s, ser):
	global availability
	x, y, w, h = -100, -100, -100, -100
	while True:
		frame = detection_buffer.get()[10:-20, 30:-20]
		if frame is None:
			break
		
		#for i in range(len(rod_slopes)):
			#cv2.line(frame, (0, int(rod_intercepts[i])), (W, int(rod_intercepts[i] + rod_slopes[i] * W)), (0, 0, 255), 2)

		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# h, s, v = cv2.split(hsv)
		# v = np.clip(v * 0.7, 0, 255).astype(np.uint8)  # reduce brightness
		# hsv = cv2.merge((h, s, v))
		mask = cv2.inRange(hsv, lower_orange, upper_orange)
		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		lateral_mask = cv2.inRange(hsv, colors_low[0], colors_high[0])
		lateral_mask = cv2.erode(lateral_mask, None, iterations=2)
		lateral_mask = cv2.dilate(lateral_mask, None, iterations=2)
		
		lateral_mask1 = cv2.inRange(hsv, colors_low[1], colors_high[1])
		lateral_mask1 = cv2.erode(lateral_mask1, None, iterations=2)
		lateral_mask1 = cv2.dilate(lateral_mask1, None, iterations=2)
		
		lateral_mask = cv2.bitwise_or(lateral_mask, lateral_mask1)

		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		max_area = 0
		dx_dt, dy_dt = 0, 0
		speed = 0
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > max_area:
				max_area = area
				x, y, w, h = cv2.boundingRect(cnt)
		if max_area > 100:
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
					#cv2.arrowedLine(frame, pos, end, (0,0,255), 2, tipLength=0.25)

					# Debug overlay
					speed = np.hypot(dx_dt, dy_dt)
					#cv2.putText(frame, f"angle:{np.degrees(angle):.1f} deg  speed:{speed:.2f}",
								#(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

		#cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

		for i in range(len(num_foosmen)):
			contours, _ = cv2.findContours(lateral_mask[:, location_start[i]:location_end[i]], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			# 2. Sort contours by area (descending)
			# We use cv2.contourArea as the key for sorting
			cnts_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

			# 3. Select the top N contours, where N is num_foosmen[i]
			# Python list slicing [:N] handles cases where len(cnts) < N automatically
			target_contours = cnts_sorted[:num_foosmen[i]]

			# 4. Loop through the selected top contours and draw
			avg_pos_this_row = 0
			for cnt in target_contours:
				x1, y1, w1, h1 = cv2.boundingRect(cnt)
				avg_pos_this_row += (y1 + h1 // 2)
				#cv2.rectangle(frame, 
					  #(x1 + location_start[i], y1), 
					  #(x1 + location_start[i] + w1, y1 + h1), 
					  #(0, 255, 0), 2)
			avg_pos_this_row //= num_foosmen[i]
			if avg_pos_this_row:
				avg_pos[i] = avg_pos_this_row
		
		kicked = -1
		intersection = None
		if speed >= 3.0:
			intersection, traj_pts = reflect_trajectory(
				pos, (dx_dt, dy_dt),
				max_bounces=5,
				line_slope=rod_slopes,
				line_intercept=rod_intercepts
			)
			# Draw the trajectory polyline
			# lengths_sum = 0
			#for i in range(len(traj_pts)-1):
				#cv2.line(frame, (int(traj_pts[i][0]), int(traj_pts[i][1])),
							#(int(traj_pts[i+1][0]), int(traj_pts[i+1][1])),
							#(255,0,0), 1)

			# Draw intersection if found
			if intersection:
				i = intersection[2]				
				if len(traj_pts) == 2:
					lengths_sum = math.sqrt((int(traj_pts[0][0]) - int(traj_pts[1][0])) ** 2 + (int(traj_pts[0][1]) - int(traj_pts[1][1])) ** 2)
					if speed:
						timed = lengths_sum / speed
				
						# Detect when to kick and simulate; this will be replaced by a kick signal in the future (but only once for the first trigger into threshold)
						if (timed < 0 and timed > -KICK_THRESHOLD) or (timed > 0 and timed < KICK_THRESHOLD):
							#cv2.line(frame, (0, int(rod_intercepts[i])), (W, int(rod_intercepts[i] + rod_slopes[i] * W)), (0, 255, 255), 5)
							kicked = i
				#cv2.circle(frame, (int(intersection[0]), int(intersection[1])),
						#6, (0,255,255), -1)
		if x > W / 2 and intersection:
			percentage = percent_along_line(rod_pt1s[i], rod_pt2s[i], intersection[:2])
		else:
			percentage = int((y + dy_dt / 40) / H * 100)

		if i == 0: # Special case: first rod
			move_dist = 0
			if percentage < min_first:
				#cv2.circle(frame, point_on_line(rod_pt1s[i], rod_pt2s[i], min_first), 5, (255, 255, 255), -1)
				move_dist = 67 * (avg_pos[0] / H - min_first / 100)
				print(avg_pos[0], move_dist)
			elif percentage > max_first:
				#cv2.circle(frame, point_on_line(rod_pt1s[i], rod_pt2s[i], max_first), 5, (255, 255, 255), -1)
				move_dist = 67 * (avg_pos[0] / H - max_first / 100)
				print(avg_pos[0], move_dist)
			else:
				#cv2.circle(frame, point_on_line(rod_pt1s[i], rod_pt2s[i], percentage), 5, (255, 255, 255), -1)
				# move_dist = 67 * (avg_pos[0] - intersection[1]) // H
				move_dist = 67 * (avg_pos[0] / H - percentage / 100)
				print(avg_pos[0], move_dist)
			if availability[1]:
				# availability[1] = False
				print(time.time())
				if move_dist >= 0:
					kick(ser, f'b1{int(move_dist):02d}')
				else:
					kick(ser, f'b0{int(-move_dist):02d}')
		else:
			partitions_this_rod = partitions[i]
			for j in range(len(partitions_this_rod)):
				if percentage < partitions_this_rod[j]:
					gap_this_rod = gaps[i]
					percentage -= gap_this_rod * j
					for _ in range(len(partitions_this_rod)):
						#cv2.circle(frame, point_on_line(rod_pt1s[i], rod_pt2s[i], percentage), 5, (255, 255, 255), -1)
						percentage += gap_this_rod
					break
		if kicked == -1:
			for i in range(len(partitions)):
				dist_to_rod = abs(rod_pt1s[i][0] - (x + w // 2))
				if dist_to_rod < LENGTH_THRESHOLD:
					#cv2.line(frame, (0, int(rod_intercepts[i])), (W, int(rod_intercepts[i] + rod_slopes[i] * W)), (0, 255, 255), 5)
					kicked = i
					break
		match kicked:
			case 0 if availability[0]:
				availability[0] = False
				kick(ser, 'a')
				
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


def exit_handler(sig, frame):
    kick(ser, 'Z')
    exit(0)


if __name__ == "__main__":
    # Set up camera backend and retrieve video capture object
    cap = init(100)

    id_to_center = process_aruco(cap, NUM_ARUCOS)
    
    cap.release()
    cap = init(30)

    H_matrix = get_perspective_matrix(id_to_center, W, H)

    rod_slopes, rod_intercepts, rod_pt1s, rod_pt2s = get_rod_lines(id_to_center, H_matrix)
    rod_slopes = rod_slopes[:2]
    rod_intercepts = rod_intercepts[:2]
    rod_pt1s = rod_pt1s[:2]
    rod_pt2s = rod_pt2s[:2]
    for i in range(len(rod_intercepts)):
        rod_intercepts[i] += rod_slopes[i] * 20
    
    ser = init_control()
    if not ser:
        print("Failed to set up serial connection.")
        exit(0)
    # --------------------
    # Multiprocessing loop
    # --------------------
    detection_buffer = Queue()
    pool = Pool(5, initializer=init_pool, initargs=(detection_buffer,))
    show_future = pool.apply_async(show, args=(rod_slopes, rod_intercepts, rod_pt1s, rod_pt2s, ser))
    receive_thread = threading.Thread(target=receive, args=(ser,))
    receive_thread.start()
    futures = []
    signal.signal(signal.SIGINT, exit_handler)

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
    receive_thread.join()

    cap.release()
    cv2.destroyAllWindows()

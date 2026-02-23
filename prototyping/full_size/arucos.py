from cv2 import findHomography, convertScaleAbs
from cv2.aruco import DICT_4X4_50, getPredefinedDictionary, DetectorParameters, ArucoDetector
import numpy as np

def process_aruco(cap, num_arucos):
    params = DetectorParameters()
    params.minMarkerPerimeterRate = 0.005
    detector = ArucoDetector(getPredefinedDictionary(DICT_4X4_50), params)

    # Detect all ArUco markers and confirm there is one for each corner and one for each end of the motorized rods
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        exit()

    corners, ids, _ = detector.detectMarkers(frame)
    new_ids = []
    new_corners = []
    if ids is not None:
        for i in range(len(ids)):
            #print(ids[i][0])
            if ids[i][0] >= 0 and ids[i][0] < 12:
                new_ids.append(ids[i])
                new_corners.append(corners[i])
    if ids is None or len(new_ids) < num_arucos:
        print(f"Not all {num_arucos} ArUco markers detected. IDs: {new_ids}")
        exit()
    
    # Find the point which each ArUco marker is pointing to
    id_to_center = {}
    for id, c in zip(new_ids, new_corners):
        if id[0] > 3:
            # Get center position of all ArUcos for rod ends
            id_to_center[id[0]] = np.mean(c[0], axis=0)
        else:
            # Top left (ArUco 0): bottom right (corner 2)
            # Top right (ArUco 1): bottom left (corner 3)
            # Bottom left (ArUco 3): top right (corner 1)
            # Bottom right (ArUco 2): top left (corner 0)
            id_to_center[id[0]] = c[0][(id[0] + 2) & 3] # Fast way to perform modulo 4
    
    return id_to_center


def get_perspective_matrix(id_to_center, W, H):
    # Check if all corners are included correctly and retrieve the desired point from each marker
    src_pts = []
    for target_id in range(4):
        if target_id not in id_to_center:
            print(f"Marker ID {target_id} not found.")
            exit()
        src_pts.append(id_to_center[target_id])
    src_pts = np.array(src_pts, dtype="float32")

    # This defines the 2D pixel plane where the video is to be projected
    dst_pts = np.array([
        [0, 0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0, H - 1]
    ], dtype="float32")

    H_matrix, _ = findHomography(src_pts, dst_pts)
    
    return H_matrix

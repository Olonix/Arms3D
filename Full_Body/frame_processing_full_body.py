import os
import cv2
import json
import numpy as np
# --- CHANGED: Removed torch and ultralytics imports ---
# import torch
# from ultralytics import YOLO
# --- CHANGED: Added mediapipe import ---
import mediapipe as mp
from Full_Body.help_modules.rotate_translate import load_plane_params, rotation_matrix_from_vectors, rotate_skeleton, translate_skeleton_to_origin
# from help_modules.frame_visualizer import visualize_3d_as_plt


# Path parameters
FRAMES_DIR = "input_color_frames"
DEPTH_DIR = "input_depth_frames"
OUTPUT_JSON_DIR = "primary_skeletons"
OUTPUT_3D_DIR = "primary_3d_vis" # PRIMARY VISUALIZATIONS FOR DEBUG !!!

# Create output folders if they don't exist
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_3D_DIR, exist_ok=True)

# Depth sampling parameter: neighborhood radius in pixels
DEPTH_SAMPLE_RADIUS = 3

# Camera settings
fx, fy = 1123.62, 1123.11
cx, cy = 942.68, 544.956
depth_scale = 1000.0

# --- Threshold values for acceptable displacement (in meters) for each point ---
DIST_THRESHOLDS = {
    # "head": 0.5,
    # "left_shoulder": 0.15,
    # "right_shoulder": 0.15,
    # "left_elbow": 0.3,
    # "right_elbow": 0.3,
    # "left_wrist": 0.5,
    # "right_wrist": 0.5,
    # "left_hip": 0.2,
    # "right_hip": 0.2,
    # "left_knee": 0.3,
    # "right_knee": 0.3,
    # "left_ankle": 1.0,
    # "right_ankle": 1.0,
}

# --- Define MediaPipe Pose landmarks and mapping to our desired names ---
# We map MediaPipe landmark indices to the names used in the rest of the code.
# Source: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#pose_landmarks
MEDIAPIPE_KEYPOINT_INDEXES = {
    0: "nose",
    1: "left_eye_inner", # Not directly used
    2: "left_eye",       # Used for head calculation
    3: "left_eye_outer", # Not directly used
    4: "right_eye_inner",# Not directly used
    5: "right_eye",      # Used for head calculation
    6: "right_eye_outer",# Not directly used
    7: "left_ear",       # Excluded
    8: "right_ear",      # Excluded
    9: "mouth_left",     # Excluded
    10: "mouth_right",    # Excluded
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",     # Excluded
    18: "right_pinky",    # Excluded
    19: "left_index",     # Excluded
    20: "right_index",    # Excluded
    21: "left_thumb",     # Excluded
    22: "right_thumb",    # Excluded
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",      # Excluded
    30: "right_heel",     # Excluded
    31: "left_foot_index",# Excluded
    32: "right_foot_index"# Excluded
}

# --- Indices to exclude based on MediaPipe numbering ---
EXCLUDED_INDICES_MP = {0, 1, 3, 4, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22, 29, 30, 31, 32}
# --- Indices for eyes based on MediaPipe numbering ---
EYE_INDICES_MP = {2, 5} # left_eye, right_eye


# --- Functions get_depth_value and image_to_3d remain ---
def get_depth_value(depth_img, u, v, radius=DEPTH_SAMPLE_RADIUS):
    """
    Non-zero values ​​are averaged around a point (u, v) in a depth image (16-bit PNG)
    If there are no non-zero values, 0 is returned
    """
    h, w = depth_img.shape
    u = int(round(u))
    v = int(round(v))

    if not (0 <= u < w and 0 <= v < h):
        return 0.0

    u_min = max(u - radius, 0)
    u_max = min(u + radius, w - 1)
    v_min = max(v - radius, 0)
    v_max = min(v + radius, h - 1)

    patch = depth_img[v_min:v_max+1, u_min:u_max+1]
    non_zero = patch[patch > 0]
    if non_zero.size == 0:
        return 0.0
    # --- Using MEDIAN ---
    return float(np.median(non_zero)) / depth_scale


def image_to_3d(u, v, depth, cx, cy, fx, fy):
    """
    Conversion from image and depth coordinates to 3D
    """
    if depth <= 0:
        return None
    x = (u - cx) * depth / fx
    y = - (v - cy) * depth / fy
    z = - depth


    return [float(x), float(y), float(z)]


# process_frame now uses MediaPipe
def process_frame(jpg_path, depth_path, pose_processor):
    """
    Processing a single frame:
    - Loading an image and a depth map
    - Detecting keypoints using MediaPipe Pose
    - Calculating depth for each keypoint (excluding specified points; using eyes for head)
    - Converting to 3D coordinates for the full body
    """
    img = cv2.imread(jpg_path)
    if img is None:
        print(f"Loading error {jpg_path}")
        return None
    h_img, w_img, _ = img.shape # Get image dimensions for de-normalization

    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print(f"Loading error {depth_path}")
        return None
    if not np.issubdtype(depth_img.dtype, np.number):
         print(f"Error: Depth map {depth_path} has non-numeric data type: {depth_img.dtype}")
         return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose_processor.process(img_rgb)

    if not results.pose_landmarks:
        print(f"Keypoints not found in {jpg_path}")
        return None

    landmarks = results.pose_landmarks.landmark
    num_detected_points = len(landmarks)

    skeleton_3d = {}

    # Uses MediaPipe indices and mapping
    for idx, landmark in enumerate(landmarks):
        if idx in EXCLUDED_INDICES_MP or idx in EYE_INDICES_MP:
            continue

        key_name = MEDIAPIPE_KEYPOINT_INDEXES.get(idx)
        if key_name is None:
            continue

        # MediaPipe provides normalized coordinates (0.0 to 1.0).
        # De-normalize to pixel coordinates.
        if landmark.visibility < 0.1:
             skeleton_3d[key_name] = None
             continue

        x = landmark.x * w_img
        y = landmark.y * h_img

        depth_val = get_depth_value(depth_img, x, y)
        pt_3d = image_to_3d(x, y, depth_val, cx, cy, fx, fy)
        skeleton_3d[key_name] = pt_3d


    left_eye_idx_mp = 2
    right_eye_idx_mp = 5
    new_head_3d = None

    # Check visibility/presence of both eyes
    if (left_eye_idx_mp < num_detected_points and
        right_eye_idx_mp < num_detected_points and
        landmarks[left_eye_idx_mp].visibility > 0.5 and
        landmarks[right_eye_idx_mp].visibility > 0.5):

        left_eye_lm = landmarks[left_eye_idx_mp]
        right_eye_lm = landmarks[right_eye_idx_mp]

        # Get coordinates from MediaPipe landmarks
        # De-normalize eye coordinates
        left_eye_x = left_eye_lm.x * w_img
        left_eye_y = left_eye_lm.y * h_img
        right_eye_x = right_eye_lm.x * w_img
        right_eye_y = right_eye_lm.y * h_img

        new_head_x_2d = (left_eye_x + right_eye_x) / 2.0
        new_head_y_2d = (left_eye_y + right_eye_y) / 2.0

        depth_head = get_depth_value(depth_img, new_head_x_2d, new_head_y_2d)

        new_head_3d = image_to_3d(new_head_x_2d, new_head_y_2d, depth_head, cx, cy, fx, fy)

    else:
        print(f"Warning: Both eyes not found/not visible enough to calculate 'head' in {jpg_path}")

    skeleton_3d["head"] = new_head_3d

    return skeleton_3d


def main_processor():
    
    if not os.path.isdir(FRAMES_DIR):
        print(f"Error: Color frames folder '{FRAMES_DIR}' not found")
    elif not os.path.isdir(DEPTH_DIR):
        print(f"Error: Depth frames folder '{DEPTH_DIR}' not found")
    else:
        for filename in os.listdir(OUTPUT_JSON_DIR):
            file_path = os.path.join(OUTPUT_JSON_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        for filename in os.listdir(OUTPUT_3D_DIR):
            file_path = os.path.join(OUTPUT_3D_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

    mp_pose = mp.solutions.pose
    # Initialize Pose model.
    # static_image_mode=False for video/sequential frames
    # jy=1 is a balance between speed and accuracy (0, 1, 2)
    # enable_segmentation=False as we don't use the mask
    # min_detection_confidence and min_tracking_confidence can be tuned
    try:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        print("MediaPipe Pose model initialized.")
    except Exception as e:
        print(f"Error initializing MediaPipe Pose: {e}")
        return

    try:
        jpg_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(".jpg")])
        if not jpg_files:
            print(f"Error: No .jpg files found in {FRAMES_DIR}")
            if 'pose' in locals() and pose: pose.close() # Close pose model if initialized
            return
    except FileNotFoundError:
        print(f"Error: Folder {FRAMES_DIR} not found.")
        if 'pose' in locals() and pose: pose.close()
        return
    except Exception as e:
        print(f"Error reading files from {FRAMES_DIR}: {e}")
        if 'pose' in locals() and pose: pose.close()
        return

    prev_skeleton = None # skeleton of the previous frame
    plane_normal = load_plane_params('floor_params.json') # normal to the floor plane

    for i, jpg_file in enumerate(jpg_files):
        jpg_path = os.path.join(FRAMES_DIR, jpg_file)
        depth_filename = f"depth_frame_{i:05d}.png"
        depth_path = os.path.join(DEPTH_DIR, depth_filename)

        if not os.path.exists(depth_path):
            print(f"Warning: Depth file not found: {depth_path}. Frame {jpg_file} skipped")
            continue
        
        if i % 100 == 0:
            print(f"Frame processing: {jpg_file} with depth: {depth_filename}")
        skeleton_3d = process_frame(jpg_path, depth_path, pose)

        if skeleton_3d is None:
            print(f"Pass {jpg_file}")
            # prev_skeleton = None # Optional: Reset smoothing on detection failure
            continue

        up = np.array([0, 0, -1], dtype=float)
        R = rotation_matrix_from_vectors(plane_normal, up)

        aligned_skel = rotate_skeleton(skeleton_3d, R)

        if i == 0:
            origin_point = np.mean([aligned_skel[i] for i in ["right_shoulder", "right_hip", "left_shoulder", "left_hip"]], axis=0) # center between shoulders and hips

        centered_skel = translate_skeleton_to_origin(aligned_skel, origin_point)

        skeleton_3d_to_save = {}
        if prev_skeleton is not None:
            # --- Determine keys based on the current map (excluding non-body parts) ---
            all_possible_keys = set(MEDIAPIPE_KEYPOINT_INDEXES.values()) - {"nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_heel", "right_heel", "left_foot_index", "right_foot_index"}
            all_possible_keys.add("head") # Add the computed head point

            for key in all_possible_keys:
                current_point = centered_skel.get(key)
                prev_point = prev_skeleton.get(key)

                if current_point is not None:
                    if prev_point is not None:
                        dist = np.linalg.norm(np.array(current_point) - np.array(prev_point))
                        threshold = DIST_THRESHOLDS.get(key, 0.5) # Default threshold if key somehow missing
                        if dist > threshold:
                            print(f"Too big offset for {key} ({dist:.2f} m > {threshold} m). Use coordinates from previous frame")
                            skeleton_3d_to_save[key] = prev_point
                        else:
                            skeleton_3d_to_save[key] = current_point
                    else:
                         skeleton_3d_to_save[key] = current_point
                elif prev_point is not None:
                    print(f"Point {key} is lost, take value from previous frame")
                    skeleton_3d_to_save[key] = prev_point
                else:
                    skeleton_3d_to_save[key] = None
        else:
             skeleton_3d_to_save = centered_skel

        prev_skeleton = {k: (v[:] if v is not None else None) for k, v in skeleton_3d_to_save.items()}

        out_json = {
            "frame": jpg_file,
            "skeleton": skeleton_3d_to_save
        }
        json_filename = os.path.splitext(jpg_file)[0] + ".json"
        json_path = os.path.join(OUTPUT_JSON_DIR, json_filename)
        try:
            # Use a custom default function to handle None and numpy types
            def json_dumper(obj):
                if isinstance(obj, (np.ndarray, np.generic)):
                    return obj.tolist()
                if obj is None:
                    return None
                # Let the default json encoder handle it
                # raise TypeError # Or let default handle it
                return json.JSONEncoder().default(obj) # Fallback

            with open(json_path, "w") as f:
                 json.dump(out_json, f, indent=4, default=json_dumper)

            if i % 100 == 0:
                print(f"Saved JSON: {json_path}")

        except Exception as e:
            print(f"Error saving JSON {json_path}: {e}")

        # --- PRIMARY VISUALIZATIONS FOR DEBUG !!! ---
        # skel_filename = os.path.splitext(jpg_file)[0]
        # skel_path = os.path.join(OUTPUT_3D_DIR, skel_filename)
        # try:
        #     visualize_3d_as_plt(skeleton_3d_to_save, skel_path)
        #     print(f"Saved 3D visualization: {skel_path}.png")
        # except Exception as e:
        #     print(f"Error creating 3D visualization for {jpg_file}: {e}")

    if 'pose' in locals() and pose:
        pose.close()
        print("MediaPipe Pose model is closed")



if __name__ == "__main__":
    main_processor()
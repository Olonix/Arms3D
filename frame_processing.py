import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# Path parameters
FRAMES_DIR = "input_frames"
DEPTH_DIR = "input_depth_frames"
OUTPUT_JSON_DIR = "output_skeletons"
OUTPUT_3D_DIR = "output_3d_vis"

# Create output folders if they don't exist
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_3D_DIR, exist_ok=True)

# Depth sampling parameter: neighborhood radius in pixels
DEPTH_SAMPLE_RADIUS = 3

# Camera settings
fx, fy = 636.6593, 636.2519
cx, cy = 635.2839, 366.8740
depth_scale = 1000.0

# Threshold values for acceptable displacement (in meters) for each point
DIST_THRESHOLDS = {
    "head": 0.5,
    "left_shoulder": 0.05,
    "right_shoulder": 0.05,
    "left_elbow": 0.3,
    "right_elbow": 0.3,
    "left_wrist": 0.5,
    "right_wrist": 0.5
}


def get_depth_value(depth_img, u, v, radius=DEPTH_SAMPLE_RADIUS):
    """
    Non-zero values ​​are averaged around a point (u, v) in a depth image (16-bit PNG).
    If there are no non-zero values, 0 is returned.
    """
    h, w = depth_img.shape
    u = int(round(u))
    v = int(round(v))
    
    u_min = max(u - radius, 0)
    u_max = min(u + radius, w - 1)
    v_min = max(v - radius, 0)
    v_max = min(v + radius, h - 1)
    
    patch = depth_img[v_min:v_max+1, u_min:u_max+1]
    non_zero = patch[patch > 0]
    if non_zero.size == 0:
        return 0
    return float(np.median(non_zero)) / depth_scale


def image_to_3d(u, v, depth, cx, cy, fx, fy):
    """
    Conversion from image and depth coordinates to 3D.
    """
    if depth == 0:
        return None
    x = (u - cx) * depth / fx
    y = - (v - cy) * depth / fy
    z = - depth
    return [x, y, z]


def visualize_3d(skel_3d, output_path):
    """
    Visualization of 3D skeleton using matplotlib library.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    points = {}
    for key, pt in skel_3d.items():
        if pt is not None:
            points[key] = pt

    for key, pt in points.items():
        ax.scatter(pt[0], pt[1], pt[2], s=50)
        # shifting names
        if key == 'right_wrist':
            ax.text(pt[0] + 0.1, pt[1], pt[2] + 0.01, key, fontsize=8)
        elif key == 'right_elbow':
            ax.text(pt[0] + 0.1, pt[1], pt[2] - 0.01, key, fontsize=8)
        else:
            ax.text(pt[0] + 0.1, pt[1], pt[2], key, fontsize=8)

    def try_line(pt1, pt2):
        if pt1 is not None and pt2 is not None:
            xs = [pt1[0], pt2[0]]
            ys = [pt1[1], pt2[1]]
            zs = [pt1[2], pt2[2]]
            ax.plot(xs, ys, zs, 'k-', linewidth=2)

    try_line(points.get("head"), points.get("left_shoulder"))
    try_line(points.get("head"), points.get("right_shoulder"))
    try_line(points.get("left_shoulder"), points.get("left_elbow"))
    try_line(points.get("left_elbow"), points.get("left_wrist"))
    try_line(points.get("right_shoulder"), points.get("right_elbow"))
    try_line(points.get("right_elbow"), points.get("right_wrist"))
    
    ax.set_xlim([-1.5, 2.0])
    ax.set_ylim([-1.25, 0.25])
    ax.set_zlim([-1.5, -0.8])
    # plt.title("3D Skeleton")

    ax.set_xticklabels([]) 
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    
    if output_path:
        plt.savefig(output_path)
    
    # plt.show()
    plt.close()


frame_idx = 1

def show_depth_with_keypoints(depth_img, keypoints):
    """
    Convert from 16-bit to 8-bit and visualize depth maps 
    with keypoint overlays defined by YOLO.
    
    If keypoints is not a dictionary, it is assumed to be a Keypoints object 
    from Ultralytics and will be converted to a dictionary of keypoint coordinates:
    - head: 0
    - left_shoulder: 5
    - right_shoulder: 6
    - left_elbow: 7
    - right_elbow: 8
    - left_wrist: 9
    - right_wrist: 10
    """

    global frame_idx

    # if keypoints is not a dict, convert it
    if not isinstance(keypoints, dict):
        keypoints_np = keypoints.cpu().numpy()[0]
        idx_map = {
            "head": 0,
            "left_shoulder": 5,
            "right_shoulder": 6,
            "left_elbow": 7,
            "right_elbow": 8,
            "left_wrist": 9,
            "right_wrist": 10
        }
        kp_dict = {}
        for key, idx in idx_map.items():
            kp_dict[key] = keypoints_np.data[0][idx][:2]  # (x, y)
        keypoints = kp_dict

    # Normalize the depth image to the range 0-255 and convert to 8-bit
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)

    # Save image to folder
    save_dir = "depth_8bit"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"depth_frame_{frame_idx:03d}.png")
    cv2.imwrite(save_path, depth_norm)
    
    depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

    # Overlay keypoints with red circles
    for key, pt in keypoints.items():
        if pt is not None:
            u, v = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(depth_bgr, (u, v), 15, (0, 0, 255), -1)  # red circle (BGR: (0,0,255))
            # cv2.putText(depth_bgr, key, (u - 5, v - 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 255), 2)

    # cv2.imshow("Depth Map with Keypoints", depth_bgr)
    # cv2.waitKey(1)

    # Save image to another folder
    save_dir = "depth_visualizations"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"depth_frame_{frame_idx:03d}.png")
    cv2.imwrite(save_path, depth_bgr)

    frame_idx += 1


def process_frame(jpg_path, depth_path, model):
    """
    Processing a single frame:
    - Loading an image and a depth map
    - Detecting keypoints using YOLO‑pose
    - Calculating depth for each keypoint
    - Converting to 3D coordinates
    """
    img = cv2.imread(jpg_path)
    if img is None:
        print(f"Loading error {jpg_path}")
        return None
    # height, width = img.shape[:2]

    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print(f"Loading error {depth_path}")
        return None

    results = model(img, verbose=False)
    if len(results) == 0 or len(results[0].keypoints) == 0:
        print(f"Keypoints not found in {jpg_path}")
        return None

    keypoints = results[0].keypoints.cpu().numpy()[0]
    idx_map = {
        "head": 0,            # initially the nose
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10
    }
    
    skeleton_3d = {}
    # First calculate 3D points for all key points by default
    for key, idx in idx_map.items():
        x, y, _ = keypoints.data[0][idx]
        depth_val = get_depth_value(depth_img, x, y)
        pt_3d = image_to_3d(x, y, depth_val, cx, cy, fx, fy)
        skeleton_3d[key] = pt_3d

    # show_depth_with_keypoints(depth_img, keypoints)

    # Recalculate the head coordinate.
    # Use the coordinates of the nose (initially keypoints[0]) and shoulders.
    nose = keypoints.data[0][0][:2]  # nose [x, y]
    left_shoulder = keypoints.data[0][5][:2]
    right_shoulder = keypoints.data[0][6][:2]
    # Calculate the middle of the shoulders
    shoulders_mid = [ (left_shoulder[0] + right_shoulder[0]) / 2.0,
                      (left_shoulder[1] + right_shoulder[1]) / 2.0 ]
    # The middle between the nose and the center of the shoulders is the new point of the head
    # (temporary solution for head detection)
    new_head_2d = [ (nose[0] + shoulders_mid[0]) / 2.0,
                    (nose[1] + shoulders_mid[1]) / 2.0 ]
    depth_head = get_depth_value(depth_img, new_head_2d[0], new_head_2d[1])
    new_head_3d = image_to_3d(new_head_2d[0], new_head_2d[1], depth_head, cx, cy, fx, fy)
    skeleton_3d["head"] = new_head_3d

    return skeleton_3d


def main():
    model = YOLO("yolo11x-pose.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    jpg_files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    
    prev_skeleton = None  # skeleton of the previous frame
    for i, jpg_file in enumerate(jpg_files):
        jpg_path = os.path.join(FRAMES_DIR, jpg_file)
        depth_filename = f"transformed_depth_image_{i+1}.png"
        depth_path = os.path.join(DEPTH_DIR, depth_filename)
        
        print(f"Frame processing: {jpg_file} with depth: {depth_filename}")
        skeleton_3d = process_frame(jpg_path, depth_path, model)
        if skeleton_3d is None:
            print(f"Pass {jpg_file}")
            continue
        
        if prev_skeleton is not None:
            for key in skeleton_3d:
                # If the point is lost in the current frame, take the value from the previous frame
                if skeleton_3d[key] is None and prev_skeleton.get(key) is not None:
                    print(f"Point {key} is lost, take value from previous frame")
                    skeleton_3d[key] = prev_skeleton[key]
                # If there is a point, check the offset relative to the previous frame
                elif skeleton_3d[key] is not None and prev_skeleton.get(key) is not None:
                    dist = np.linalg.norm(np.array(skeleton_3d[key]) - np.array(prev_skeleton[key]))
                    if dist > DIST_THRESHOLDS[key]:
                        print(f"Too big offset for {key} ({dist:.2f} m). Use coordinates from previous frame")
                        skeleton_3d[key] = prev_skeleton[key]
        
        # Save the current skeleton for the next iteration
        prev_skeleton = {k: (v[:] if v is not None else None) for k, v in skeleton_3d.items()}
        
        out_json = {
            "frame": jpg_file,
            "skeleton": skeleton_3d
        }
        json_filename = os.path.splitext(jpg_file)[0] + ".json"
        json_path = os.path.join(OUTPUT_JSON_DIR, json_filename)
        with open(json_path, "w") as f:
            json.dump(out_json, f, indent=4)
        
        skel_filename = os.path.splitext(jpg_file)[0]
        skel_path = os.path.join(OUTPUT_3D_DIR, skel_filename)
        visualize_3d(skeleton_3d, skel_path)

        print(f"Saved: {json_path}")

if __name__ == "__main__":
    main()
import os
import json
import numpy as np
import scipy.signal
from frame_processing import visualize_3d


INPUT_JSON_DIR = "output_skeletons"
OUTPUT_JSON_DIR = "smoothed_skeletons"
OUTPUT_VIS_DIR = "smoothed_3d_vis"

# Butterworth filter parameters
FILTER_ORDER = 5
CUTOFF_FREQ = 1.0

# Create output folders if they don't exist
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

def butter_lowpass_filter(data, cutoff=CUTOFF_FREQ, fs=30.0, order=FILTER_ORDER):
    """
    Apply a Butterworth low-pass filter to the data.
    - data: list of values along one axis for all frames
    - cutoff: cutoff frequency
    - fs: sample rate (frames per second)
    - order: filter order
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype="low", analog=False)
    if len(data) == 0 or all(np.isnan(data)):
        return data  # If there is no data or all are NaN, return as is
    return scipy.signal.filtfilt(b, a, data, axis=0)

def process_skeletons():
    """
    Read JSON files, smooth keypoint coordinates and save new JSONs.
    """
    json_files = sorted([f for f in os.listdir(INPUT_JSON_DIR) if f.endswith(".json")])

    skeleton_data = {}

    for json_file in json_files:
        json_path = os.path.join(INPUT_JSON_DIR, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)

        # frame_name = data["frame"]
        skeleton = data["skeleton"]

        for key, pt in skeleton.items():
            if pt is not None:
                if key not in skeleton_data:
                    skeleton_data[key] = {"x": [], "y": [], "z": []}
                skeleton_data[key]["x"].append(pt[0])
                skeleton_data[key]["y"].append(pt[1])
                skeleton_data[key]["z"].append(pt[2])
            else:
                if key not in skeleton_data:
                    skeleton_data[key] = {"x": [], "y": [], "z": []}
                skeleton_data[key]["x"].append(np.nan)
                skeleton_data[key]["y"].append(np.nan)
                skeleton_data[key]["z"].append(np.nan)

    smoothed_skeleton_data = {}
    for key, coords in skeleton_data.items():
        smoothed_skeleton_data[key] = {
            "x": butter_lowpass_filter(coords["x"]),
            "y": butter_lowpass_filter(coords["y"]),
            "z": butter_lowpass_filter(coords["z"]),
        }

    for i, json_file in enumerate(json_files):
        smoothed_skeleton = {}
        for key in smoothed_skeleton_data.keys():
            x, y, z = (
                smoothed_skeleton_data[key]["x"][i],
                smoothed_skeleton_data[key]["y"][i],
                smoothed_skeleton_data[key]["z"][i],
            )
            if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                smoothed_skeleton[key] = [float(x), float(y), float(z)]
            else:
                smoothed_skeleton[key] = None

        smoothed_json_path = os.path.join(OUTPUT_JSON_DIR, json_file)
        with open(smoothed_json_path, "w") as f:
            json.dump({"frame": json_file, "skeleton": smoothed_skeleton}, f, indent=4)

        vis_path = os.path.join(OUTPUT_VIS_DIR, json_file.replace(".json", "_smoothed.png"))
        visualize_3d(smoothed_skeleton, vis_path)

if __name__ == "__main__":
    process_skeletons()
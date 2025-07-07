import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

INPUT_JSON_DIR = "smoothed_skeletons"
OUTPUT_METRICS_DIR = "output_metrics"
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)


def compute_angle(v1, v2):
    """Calculate the angle between two vectors in degrees."""
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_theta = np.clip(dot_product / norms, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def get_joint_vector(pt1, pt2):
    """Get a vector between two points."""
    return np.array(pt2) - np.array(pt1)

def get_points(skeleton):
    """Extract point coordinates from JSON."""
    return {key: np.array(val) if val else None for key, val in skeleton.items()}


# Lists for storing metrics
elbow_angles = []
# wrist_distances = []
shoulder_vectors_angles = []
shoulder_tilt_angles = []
shoulder_vertical_angles = []

json_files = sorted([f for f in os.listdir(INPUT_JSON_DIR) if f.endswith(".json")])

for i, json_file in enumerate(json_files):
    with open(os.path.join(INPUT_JSON_DIR, json_file)) as f:
        data = json.load(f)
    
    skeleton = data.get("skeleton", {})
    points = get_points(skeleton)

    # Elbow flexion angle
    v1 = get_joint_vector(points["left_shoulder"], points["left_elbow"])
    v2 = get_joint_vector(points["left_elbow"], points["left_wrist"])
    elbow_angle = compute_angle(v1, v2)
    elbow_angles.append(elbow_angle)

    # # Euclidean distance between wrists
    # if points["left_wrist"] is not None and points["right_wrist"] is not None:
    #     wrist_distance = np.linalg.norm(np.array(points["left_wrist"]) - np.array(points["right_wrist"]))
    #     wrist_distances.append(wrist_distance*0.5)
    
    # Angle Between Shoulders
    if points["left_shoulder"] is not None and points["right_shoulder"] is not None:
        v3 = get_joint_vector(points["left_shoulder"], points["left_elbow"])
        v4 = get_joint_vector(points["right_shoulder"], points["right_elbow"])
        shoulder_angle = compute_angle(v3, v4)
        shoulder_vectors_angles.append(shoulder_angle)
    
    # Shoulder Tilt
    vertical_vector = np.array([0, 0, -1])  # Vertical
    shoulder_tilt = compute_angle(v1, vertical_vector)
    shoulder_tilt_angles.append(90 - shoulder_tilt)

    # Vector Connecting the Shoulders
    if points["left_shoulder"] is not None and points["right_shoulder"] is not None:
        shoulder_vector = points["right_shoulder"] - points["left_shoulder"]
        vertical_vector = np.array([0, 0, -1])
        shoulder_vertical_angle = compute_angle(shoulder_vector, vertical_vector)
        shoulder_vertical_angles.append(90 - shoulder_vertical_angle)



fps = 30
time = np.arange(len(elbow_angles)) / fps

def plot_metric(values, title, ylabel, filename, minima_x=None, interval_minima=None, show_interval_points=False):
    plt.figure()
    plt.plot(time, values, color="blue", zorder=1)
    # plt.xlabel("Time (s)", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=19)
    plt.xlim(0, time[-1])
    plt.margins(x=0)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Define the boundaries along the Y axis and add up to a whole division
    yticks = ax.get_yticks()
    if len(yticks) > 0:
        y_min, y_max = min(values), max(values)
        y_step = yticks[1] - yticks[0]
        
        y_min_aligned = y_step * np.floor(y_min / y_step)
        y_max_aligned = y_step * np.ceil(y_max / y_step)
        
        ax.set_ylim(y_min_aligned, y_max_aligned)

    # Draw vertical lines for the selected minimums
    if minima_x is not None:
        if title == "Angle Between Shoulders":
            for x in minima_x[1:]:
                plt.axvline(x, color="gray", linestyle="--", linewidth=0.8, zorder=2)
        else:
            for x in minima_x:
                plt.axvline(x, color="gray", linestyle="--", linewidth=0.8, zorder=2)

    # Draw red dots
    if show_interval_points and interval_minima is not None:
        for x, y in interval_minima:
            plt.scatter(x, y, color="red", s=30, zorder=3)  # zorder=3, so that the dots are on the top layer

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_METRICS_DIR, filename), bbox_inches='tight', pad_inches=0)
    plt.close()


def get_intervals(minima_x, time):
    """Define intervals, including a start and end interval."""
    intervals = []
    intervals.append((time[0], minima_x[0]))  # Start interval
    for i in range(len(minima_x) - 1):
        start, end = minima_x[i], minima_x[i + 1]
        intervals.append((start, end))
    intervals.append((minima_x[-1], time[-1]))  # End interval
    return intervals


def find_lowest_point(values, interval_time):
    """Return the lowest point in the range if there are no obvious minima."""
    min_idx = np.argmin(values)
    return interval_time[min_idx], values[min_idx]


def find_interval_minima(values, intervals, time, prominence=0.1):
    """Find the lowest minimum in each interval, if not, takes the lowest point."""
    interval_minima = []
    for start, end in intervals:
        mask = (time >= start) & (time <= end)
        interval_time = time[mask]
        interval_values = np.array(values)[mask]

        min_indices, _ = find_peaks(-interval_values, prominence=prominence)

        if len(min_indices) > 0:
            min_idx = min_indices[np.argmin(interval_values[min_indices])]
        else:
            min_idx = np.argmin(interval_values)

        interval_minima.append((interval_time[min_idx], interval_values[min_idx]))

    return interval_minima


# === Find the main minimums on the chart shoulder_vectors_angles ===
inverted_angles = -np.array(shoulder_vectors_angles)
min_indices, _ = find_peaks(inverted_angles, prominence=2.5)
minima_x = time[min_indices][1:]

# === Define intervals, including start and end ===
intervals = get_intervals(minima_x, time)

# === Looking for the lowest minimums for all charts by intervals ===
elbow_minima = find_interval_minima(elbow_angles, intervals, time)
# wrist_minima = find_interval_minima(wrist_distance, intervals, time)
shoulder_tilt_minima = find_interval_minima(shoulder_tilt_angles, intervals, time)
shoulder_vertical_minima = find_interval_minima(shoulder_vertical_angles, intervals, time)

# === Plot all the graphs with these minimums ===
plot_metric(elbow_angles, "Elbow Flexion", "Angle (degrees)", "elbow_angle.svg", minima_x, elbow_minima, show_interval_points=True)
# plot_metric(wrist_distances, "Distance Between Wrists", "Distance (m)", "wrist_distance.png")
plot_metric(shoulder_vectors_angles, "Angle Between Shoulders", "Angle (degrees)", "shoulder_vector_angle.svg", time[min_indices],
    [(x, shoulder_vectors_angles[list(time).index(x)]) for x in time[min_indices]],
    show_interval_points=True
)
plot_metric(shoulder_tilt_angles, "Shoulder Tilt", "Angle (degrees)", "left_shoulder_tilt.svg", minima_x, shoulder_tilt_minima, show_interval_points=True)
plot_metric(shoulder_vertical_angles, "Uneven Shoulders", "Angle (degrees)", "shoulder_vertical_angle.svg", minima_x, shoulder_vertical_minima, show_interval_points=True)

# Statistics output
print(f"Maximum Elbow Flexion Angle: {max(elbow_angles):.2f} degrees")
print(f"Minimum Elbow Flexion Angle: {min(elbow_angles):.2f} degrees")
# print(f"Minimum Distance Between Wrists: {min(wrist_distances):.2f} m")
print(f"Maximum Angle Between Shoulders: {max(shoulder_vectors_angles):.2f} degrees")
print(f"Minimum Angle Between Shoulders: {min(shoulder_vectors_angles):.2f} degrees")
print(f"Maximum Shoulder Tilt Angle: {max(shoulder_tilt_angles):.2f} degrees")
print(f"Minimum Shoulder Tilt Angle: {min(shoulder_tilt_angles):.2f} degrees")
print(f"Maximum Uneven Shoulders Angle: {max(shoulder_vertical_angles):.2f} degrees")
print(f"Minimum Uneven Shoulders Angle: {min(shoulder_vertical_angles):.2f} degrees")


# Saving results to JSON
metrics_json = {
    "max_elbow_angle": max(elbow_angles),
    "min_elbow_angle": min(elbow_angles),
    # "min_wrist_distance": min(wrist_distances),
    "max_shoulder_vector_angle": max(shoulder_vectors_angles),
    "min_shoulder_vector_angle": min(shoulder_vectors_angles),
    "max_shoulder_tilt_angle": max(shoulder_tilt_angles),
    "min_shoulder_tilt_angle": min(shoulder_tilt_angles),
    "max_shoulder_vertical_angle": max(shoulder_vertical_angles),
    "min_shoulder_vertical_angle": min(shoulder_vertical_angles)
}

with open(os.path.join(OUTPUT_METRICS_DIR, "metrics_summary.json"), "w") as f:
    json.dump(metrics_json, f, indent=4)

print(f"Metrics are saved in {OUTPUT_METRICS_DIR}")
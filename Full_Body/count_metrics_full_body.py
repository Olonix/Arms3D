import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Configuration
# top_dir = os.path.dirname(__file__)
top_dir = ""
INPUT_JSON_DIR = os.path.join(top_dir, "ekf_skeletons")
OUTPUT_METRICS_DIR = os.path.join(top_dir, "output_metrics")
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

# Parameters
FPS = 30
ACTIVE_LEG_SIDE = "left"  # "left" or "right"
# Interval for finding n local minima on leg lift plot (seconds)
MINIMA_INTERVAL = (2, 30)
# Peak detection settings
PEAK_PROMINENCE = 4.0  # adjust as needed

# Vectors
def compute_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """Return angle between v1 and v2 in degrees."""
    dot = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_t = np.clip(dot / norms, -1.0, 1.0)
    return np.degrees(np.arccos(cos_t))


def get_joint_vector(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Vector from p1 to p2."""
    return np.array(p2) - np.array(p1)


def get_points(skel: dict) -> dict:
    """Convert skeleton JSON values to numpy arrays. """
    return {k: np.array(v) if v is not None else None for k, v in skel.items()}

def get_intervals(boundaries, time):
    """Return list of (start, end) intervals given sorted boundary times."""
    intervals = []
    # before first
    intervals.append((time[0], boundaries[0]))
    for i in range(len(boundaries) - 1):
        intervals.append((boundaries[i], boundaries[i+1]))
    intervals.append((boundaries[-1], time[-1]))
    return intervals


def leg_movement_analysis():
    # Storage for metrics
    lift_angles = []
    knee_angles = []
    abduction_angles = []
    joint_positions = []

    # Load and process
    json_files = sorted([f for f in os.listdir(INPUT_JSON_DIR) if f.endswith('.json')])
    for fname in json_files:
        path = os.path.join(INPUT_JSON_DIR, fname)
        with open(path, 'r') as f:
            data = json.load(f)

        pts = get_points(data.get('skeleton', {}))

        hip = pts.get(f"{ACTIVE_LEG_SIDE}_hip")
        knee = pts.get(f"{ACTIVE_LEG_SIDE}_knee")
        ankle = pts.get(f"{ACTIVE_LEG_SIDE}_ankle")
        l_sh = pts.get('left_shoulder')
        r_sh = pts.get('right_shoulder')
        head = pts.get('head')
        l_hip = pts.get('left_hip')
        r_hip = pts.get('right_hip')

        # Skip if any joint missing
        if hip is None or knee is None or ankle is None:
            lift_angles.append(np.nan)
            knee_angles.append(np.nan)
            abduction_angles.append(np.nan)
            continue

        # 1) Amplitude of leg lift relative to floor
        # Hip vector: hip -> knee
        v_hip = get_joint_vector(hip, knee)
        # Floor normal vector (z axis up/down)
        vertical = np.array([0, 0, 1])
        # Angle relative to floor = 90deg - angle between leg and vertical
        lift_angle = 90 - compute_angle(v_hip, vertical)
        lift_angles.append(lift_angle)

        # 2) Knee flexion angle between thigh and shin
        v_thigh = get_joint_vector(hip, knee)
        v_shin = get_joint_vector(ankle, knee)
        knee_angle = compute_angle(v_thigh, v_shin)
        knee_angles.append(knee_angle)

        # 3) Abduction from sagittal plane using dynamic normal
        # Leg vector: hip -> ankle
        v_leg = get_joint_vector(hip, ankle)
        # Midpoints of shoulders and hips   
        hip_mid = (pts['left_hip'] + pts['right_hip']) / 2
        sh_mid = (l_sh + r_sh) / 2
        # Spine vector
        spine_vec = hip_mid - sh_mid
        # Project to horizontal plane
        spine_horiz = spine_vec.copy()
        spine_horiz[2] = 0
        # Compute perpendicular (normal) in horizontal plane
        norm_vec = np.array([-spine_horiz[1], spine_horiz[0], 0])
        if np.linalg.norm(norm_vec) == 0:
            abduction_angles.append(np.nan)
        else:
            abduction = compute_angle(v_leg, norm_vec) - 90
            abduction_angles.append(abduction)

        # collect joint positions for stability
        joint_positions.append([head * 100, l_sh * 100, r_sh * 100, l_hip * 100, r_hip * 100])
    
    # compute stability_rms per frame using global joint means
    arr = []
    for frame in joint_positions:
        if any(p is None for p in frame):
            arr.append([np.nan]*5)
        else:
            arr.append([p for p in frame])
    # shape (n_frames,5,3)
    stack = np.array(arr, dtype=float)
    # mean per joint over time: (5,3)
    mean_joints = np.nanmean(stack, axis=0)
    # RMS deviation per frame
    stability_rms = []
    for f in range(stack.shape[0]):
        frame = stack[f]
        if np.isnan(frame).any():
            stability_rms.append(np.nan)
        else:
            diffs = frame - mean_joints
            sq = np.sum(diffs**2, axis=1)
            stability_rms.append(np.sqrt(np.mean(sq)))

    # Time axis
    # time = np.arange(len(lift_angles)) / FPS
    time = np.arange(len(knee_angles)) / FPS

    # 1) Find n minima boundaries on lift plot in MINIMA_INTERVAL
    mask = (time >= MINIMA_INTERVAL[0]) & (time <= MINIMA_INTERVAL[1])
    time_int = time[mask]
    # vals_int = -np.array(lift_angles)[mask]
    vals_int = -np.array(knee_angles)[mask]
    peaks, _ = find_peaks(vals_int, prominence=PEAK_PROMINENCE)

    minima_times = []
    for peak in peaks:
        if vals_int[peak] > -120:
            minima_times.append(time_int[peak])

    # 2) Define n+1 intervals
    intervals = get_intervals(minima_times, time)

    # 3) Within each interval find:
    #   - lift max (one per interval)
    #   - knee min
    #   - abduction min
    grid = []
    lift_peaks = []
    # knee_troughs = []
    knee_peaks = []
    abd_troughs = []
    for start, end in intervals:
        mask_i = (time >= start) & (time <= end)
        t_i = time[mask_i]
        lift_i = np.array(lift_angles)[mask_i]
        knee_i = np.array(knee_angles)[mask_i]
        abd_i = np.array(abduction_angles)[mask_i]

        # lift maxima
        p_lift, _ = find_peaks(lift_i, prominence=PEAK_PROMINENCE)
        if len(p_lift) > 0:
            idx = p_lift[np.argmax(lift_i[p_lift])]
        else:
            idx = np.nanargmax(lift_i)
        lift_peaks.append((t_i[idx], lift_i[idx]))

        # # knee minima
        # inv_k = -knee_i
        # p_knee, _ = find_peaks(inv_k, prominence=PEAK_PROMINENCE)
        # if len(p_knee) > 0:
        #     # idxk = p_knee[np.argmin(knee_i[p_knee])]
        #     idxk = p_knee[np.argmin(knee_i[p_knee])]
        # else:
        #     idxk = np.nanargmin(knee_i)
        # knee_troughs.append((t_i[idxk], knee_i[idxk]))

        # knee maxima
        p_knee, _ = find_peaks(knee_i, prominence=PEAK_PROMINENCE)
        if len(p_knee) > 0:
            # idxk = p_knee[np.argmin(knee_i[p_knee])]
            idxk = p_knee[np.argmax(knee_i[p_knee])]
        else:
            idxk = np.nanargmin(knee_i)
        knee_peaks.append((t_i[idxk], knee_i[idxk]))

        # abd minima
        inv_a = -abd_i
        p_abd, _ = find_peaks(inv_a, prominence=PEAK_PROMINENCE)
        if len(p_abd) > 0:
            idxa = p_abd[np.argmin(abd_i[p_abd])]
        else:
            idxa = np.nanargmin(abd_i)
        abd_troughs.append((t_i[idxa], abd_i[idxa]))

    # Enhanced plotting function with custom y-limits
    def plot_metric(time, values, title, ylabel, filename,
                    y_limits=None, draw_zero=False, shade_range=None, 
                    vlines=None, points=None):
        # fig, ax = plt.subplots(figsize=(12, 5))
        fig, ax = plt.subplots()
        ax.plot(time, values, color='blue')
        # ax.set_xlabel('Time (s)', fontsize=16)
        # ax.set_ylabel(ylabel, fontsize=16)
        # ax.set_title(title, fontsize=19)
        ax.set_xlim(0, time[-30])
        ax.tick_params(axis='both', which='major', labelsize=18)

        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(2)

        # ax.grid(True, color='gray', linewidth=1)

        if y_limits is not None:
            ax.set_ylim(y_limits)
        if shade_range is not None:
            for shade in shade_range:
                ax.axhspan(shade[0], shade[1], color='darkgray', alpha=1.0)
        if draw_zero:
            ax.axhline(0, color='black', linewidth=2)
        if vlines is not None:
            for x in vlines:
                ax.axvline(x, color='black', linestyle='--', linewidth=2)
        if points:
            xs, ys = zip(*points)
            ax.scatter(xs, ys, color='red', s=40, zorder=3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_METRICS_DIR, filename), dpi=300)
        plt.close(fig)


    # Define custom y-axis limits and shaded (acceptable) ranges (in degrees)
    config = {
        'leg_lift_angle.png': {
            # 'y_limits': (-10, 50),
            # 'shade_range': None
            'y_limits': (0, 50),
            # 'shade_range': (35, 50)
            'shade_range': [(0, 35)]
        },
        'knee_flexion_angle.png': {
            'y_limits': (0, 180),
            'shade_range': [(0, 160)]
            # 'shade_range': None
        },
        'leg_abduction_angle.png': {
            'y_limits': (-45, 45),
            # 'shade_range': (-20, 20)
            'shade_range': [(-45, -20), (20, 45)]
        },
        'stability_rms.png': {
            'y_limits': (0, 5), 
            # 'shade_range': (0, 2)
            'shade_range': [(2, 5)]
        }
    }

    # Generate plots
    plot_metric(time, lift_angles, 'Leg Lift', 
                                   'Angle (deg)', 
                                   'leg_lift_angle.svg',
                                   vlines=minima_times,
                                #    points=lift_peaks,
                                   **config['leg_lift_angle.png'])
    plot_metric(time, knee_angles, 'Knee Flexion Amplitude', 
                                   'Angle (deg)', 
                                   'knee_flexion_angle.svg',
                                   vlines=minima_times,
                                #    points=knee_troughs,
                                   points=knee_peaks,
                                   **config['knee_flexion_angle.png'])
    plot_metric(time, abduction_angles, 'Leg Abduction', 
                                        'Angle (deg)', 
                                        'leg_abduction_angle.svg',
                                        draw_zero=True,
                                        vlines=minima_times,
                                        points=abd_troughs,
                                        **config['leg_abduction_angle.png'])
    plot_metric(time, stability_rms, 'Stability RMS', 
                                     'RMS deviation (m)', 
                                     'stability_rms.svg',
                                     vlines=minima_times,
                                     **config['stability_rms.png'])


    # Save summary statistics
    metrics_summary = {
        'leg_lift_angle': {
            'min': float(np.nanmin(lift_angles)),
            'max': float(np.nanmax(lift_angles)),
            'mean': float(np.nanmean(lift_angles)),
            'std': float(np.nanstd(lift_angles))
        },
        'knee_flexion_angle': {
            'min': float(np.nanmin(knee_angles)),
            'max': float(np.nanmax(knee_angles)),
            'mean': float(np.nanmean(knee_angles)),
            'std': float(np.nanstd(knee_angles))
        },
        'leg_abduction_angle': {
            'min': float(np.nanmin(abduction_angles)),
            'max': float(np.nanmax(abduction_angles)),
            'mean': float(np.nanmean(abduction_angles)),
            'std': float(np.nanstd(abduction_angles))
        },
        'stability_rms': {
            'min': float(np.nanmin(stability_rms)),
            'max': float(np.nanmax(stability_rms)),
            'mean': float(np.nanmean(stability_rms)),
            'std': float(np.nanstd(stability_rms))
        }
    }

    with open(os.path.join(OUTPUT_METRICS_DIR, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"Plots saved in '{OUTPUT_METRICS_DIR}', summary metrics in 'metrics_summary.json'.")


if __name__ == "__main__":
    leg_movement_analysis()
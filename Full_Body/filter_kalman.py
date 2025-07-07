import os
import json
import numpy as np
import re
from filterpy.kalman import ExtendedKalmanFilter
from Full_Body.help_modules.frame_visualizer import visualize_3d_as_plt
# from help_modules.fobbiden_angles import clamp_angle_and_rotate

# define the structure of the skeleton
JOINT_NAMES = [
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
]

# pairs of bones for length control
BONE_PAIRS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle")
]

HUMAN_HEIGHT = 1.6 # avg 1.6294 for women (https://multisite.eos.ncsu.edu/www-ergocenter-ncsu-edu/wp-content/uploads/sites/18/2016/06/Anthropometric-Detailed-Data-Tables.pdf#:~:text=31,67)

# HYPOTHESIS: the ratio of bone lengths to human height is the same for all women
# (for men too, but they have their own proportions)
# (https://multisite.eos.ncsu.edu/www-ergocenter-ncsu-edu/wp-content/uploads/sites/18/2016/06/Anthropometric-Detailed-Data-Tables.pdf#:~:text=31,67)
# for women
REFERENCE_BONE_LENGTHS = {
    ("left_shoulder", "left_elbow"): HUMAN_HEIGHT / 5.22,
    # ("left_shoulder", "left_elbow"): HUMAN_HEIGHT / 4.85,

    ("left_elbow", "left_wrist"): HUMAN_HEIGHT / 6.7,

    ("right_shoulder", "right_elbow"): HUMAN_HEIGHT / 5.22,
    # ("right_shoulder", "right_elbow"): HUMAN_HEIGHT / 4.85,

    ("right_elbow", "right_wrist"): HUMAN_HEIGHT / 6.7,

    ("left_hip", "left_knee"): HUMAN_HEIGHT / 4,
    # ("left_hip", "left_knee"): HUMAN_HEIGHT / 3.38,

    ("left_knee", "left_ankle"): HUMAN_HEIGHT / 4.6,
    # ("left_knee", "left_ankle"): HUMAN_HEIGHT / 4.18,

    ("right_hip", "right_knee"): HUMAN_HEIGHT / 4, 
    # ("right_hip", "right_knee"): HUMAN_HEIGHT / 3.38,

    ("right_knee", "right_ankle"): HUMAN_HEIGHT / 4.6, #4.375
    # ("right_knee", "right_ankle"): HUMAN_HEIGHT / 4.18,
}


JOINT_TRIPLES = {
    # joint name: (point_1, point_middle, point_2)
    "left_elbow":   ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow":  ("right_shoulder","right_elbow","right_wrist"),
    "left_knee":    ("left_hip",      "left_knee",  "left_ankle"),
    "right_knee":   ("right_hip",     "right_knee", "right_ankle"),
}


# calculate indices in the state vector
N = len(JOINT_NAMES)
joint_idx = {name: i for i, name in enumerate(JOINT_NAMES)}


def make_ekf(dt=1/30., var_pos=1e-2, var_len=1e-3):
    """
    Create an EKF that has:
    - dim_x = 6*N (x,y,z + vx,vy,vz for each joint)
    - dim_z = 3*N + M (M = number of bones - length measurements)
    """
    M = len(BONE_PAIRS)
    ekf = ExtendedKalmanFilter(dim_x=6*N, dim_z=3*N + M)

    # --- F: constant speed for each joint ---
    ekf.F = np.zeros((6*N, 6*N))
    for i in range(N):
        # positional part
        ekf.F[3*i+0, 3*i+0] = 1
        ekf.F[3*i+1, 3*i+1] = 1
        ekf.F[3*i+2, 3*i+2] = 1
        # speed affects position
        ekf.F[3*i+0, 3*N + 3*i+0] = dt
        ekf.F[3*i+1, 3*N + 3*i+1] = dt
        ekf.F[3*i+2, 3*N + 3*i+2] = dt
        # the speed is const
        ekf.F[3*N + 3*i+0, 3*N + 3*i+0] = 1
        ekf.F[3*N + 3*i+1, 3*N + 3*i+1] = 1
        ekf.F[3*N + 3*i+2, 3*N + 3*i+2] = 1

    # --- Q & R: measurement and process spread ---
    ekf.Q = np.eye(6*N) * var_pos      # smooth out the acceleration a little
    ekf.R = np.eye(3*N + M)
    ekf.R[:3*N, :3*N] *= var_pos       # for positions
    ekf.R[3*N:, 3*N:] *= var_len       # for bone lengths

    # initial P
    ekf.P = np.eye(6*N) * 1.0

    return ekf


def h(x):
    """
    Nonlinear measurement function:
    - first 3N outputs are just positions of each joint
    - next M are distances along BONE_PAIRS
    """
    zs = []
    # 1) positions
    for i in range(N):
        zs.extend(x[3*i:3*i+3])
    # 2) bone lengths
    for (a, b) in BONE_PAIRS:
        ia = joint_idx[a]
        ib = joint_idx[b]
        pa = x[3*ia:3*ia+3]
        pb = x[3*ib:3*ib+3]
        zs.append(np.linalg.norm(pb-pa))
    return np.array(zs)


def H_jacobian(x):
    """
    Jacobian h(x): matrix (dim_z x dim_x)
    """
    H = np.zeros((3*N + len(BONE_PAIRS), 6*N))
    # 1) derivatives by positions (=1)
    for i in range(N):
        H[3*i+0, 3*i+0] = 1
        H[3*i+1, 3*i+1] = 1
        H[3*i+2, 3*i+2] = 1

    # 2) bone lengths d(norm(pb-pa))/d(pa,pb)
    offset = 3*N
    for k, (a, b) in enumerate(BONE_PAIRS):
        ia, ib = joint_idx[a], joint_idx[b]
        pa = x[3*ia:3*ia+3]
        pb = x[3*ib:3*ib+3]
        diff = pb - pa
        norm = np.linalg.norm(diff) + 1e-9
        grad = diff / norm
        H[offset + k, 3*ia:3*ia+3] = -grad
        H[offset + k, 3*ib:3*ib+3] =  grad

    return H


def apply_ekf():
    
    INPUT_JSON_DIR  = "smoothed_skeletons"
    OUTPUT_JSON_DIR = "ekf_skeletons"
    OUTPUT_3D_DIR  = "ekf_3d_vis"

    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_3D_DIR, exist_ok=True)

    for filename in os.listdir(OUTPUT_JSON_DIR):
        file_path = os.path.join(OUTPUT_JSON_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for filename in os.listdir(OUTPUT_3D_DIR):
        file_path = os.path.join(OUTPUT_3D_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    ekf = make_ekf()
    first_frame = sorted(os.listdir(INPUT_JSON_DIR))[0]
    first_data = json.load(open(os.path.join(INPUT_JSON_DIR, first_frame)))
    x0 = np.zeros(6*N)
    for j, name in enumerate(JOINT_NAMES):
        pt = first_data["skeleton"].get(name)
        if pt is not None:
            x0[3*j:3*j+3] = pt
    ekf.x = x0

    R_initial = ekf.R.copy()

    for fname in sorted(os.listdir(INPUT_JSON_DIR)):
        data = json.load(open(os.path.join(INPUT_JSON_DIR, fname)))

        # 1) Predict
        ekf.predict()

        # 2) Form z of fixed length
        dim_z = 3*N + len(BONE_PAIRS)
        z = np.zeros(dim_z)

        # a) positions of joints
        for i, name in enumerate(JOINT_NAMES):
            idx = 3*i
            pt = data["skeleton"].get(name)
            if pt is None:
                # we put the predicted and lower the trust
                z[idx:idx+3] = ekf.x[idx:idx+3]
                ekf.R[idx,idx]     = 1e6
                ekf.R[idx+1,idx+1] = 1e6
                ekf.R[idx+2,idx+2] = 1e6
            else:
                z[idx:idx+3] = pt

        # b) bone lengths
        offset = 3*N
        for k, (j1, j2) in enumerate(BONE_PAIRS):
            z_idx = offset + k
            p1 = data["skeleton"].get(j1)
            p2 = data["skeleton"].get(j2)
            if p1 is None or p2 is None:
                # substitute the predicted distance
                diff = ekf.x[3*joint_idx[j2]:3*joint_idx[j2]+3] \
                     - ekf.x[3*joint_idx[j1]:3*joint_idx[j1]+3]
                z[z_idx] = np.linalg.norm(diff)
                ekf.R[z_idx, z_idx] = 1e6
            else:
                key = (j1,j2) if (j1,j2) in REFERENCE_BONE_LENGTHS else (j2,j1)
                z[z_idx] = REFERENCE_BONE_LENGTHS[key]

        # 3) Update
        ekf.update(z, HJacobian=H_jacobian, Hx=h)

        # 4) Restore R
        ekf.R = R_initial.copy()

        # 5) Unpack the adjusted positions
        corrected = {}
        for j, name in enumerate(JOINT_NAMES):
            if data["skeleton"].get(name) is None:
                corrected[name] = None
            else:
                corrected[name] = ekf.x[3*j:3*j+3].tolist()

        # === NEW PART: Forbidden Angles ===
        # for joint, (A, B, C) in JOINT_TRIPLES.items():
        #     if corrected.get(A) and corrected.get(B) and corrected.get(C):
        #         corrected[C] = clamp_angle_and_rotate(corrected, A, B, C)
        #         # if corrected_C is not None:
        #         #     corrected[C] = corrected_C
        # === end of new part ===

        # for debug
        # print(np.linalg.norm(np.array(corrected["left_knee"]) - np.array(corrected["left_ankle"])))

        # 6) Save and visualize
        match = re.search(r'(\d+)', fname).group(1)
        out = {"frame": match, "skeleton": corrected}
        with open(os.path.join(OUTPUT_JSON_DIR, OUTPUT_JSON_DIR + "-EKF-SKEL-" + match + ".json"), "w") as f:
            json.dump(out, f, indent=4)

        vis_path = os.path.join(OUTPUT_3D_DIR, fname.replace(".json", "_ekf"))
        visualize_3d_as_plt(corrected, vis_path)


if __name__ == "__main__":
    apply_ekf()
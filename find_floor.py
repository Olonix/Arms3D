import open3d as o3d
import numpy as np
from sklearn.linear_model import LinearRegression
import json
import subprocess


# Path to video
VIDEO_PATH = "D:\OrbbecSDK_K4A_Wrapper\include\output_all_color_params_2025-04-25_16-03-07.mkv"  # replace with your .mkv file path


def fit_plane_linear_regression(ply_file, output_json):
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    X = points[:, :2]  # x, y
    y = points[:, 2]   # z

    # z = ax + by + c
    model = LinearRegression()
    model.fit(X, y)

    a, b = model.coef_
    c = model.intercept_

    A = a
    B = b
    C = -1
    D = c

    plane_params = {
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "D": float(D)
    }

    with open(output_json, 'w') as f:
        json.dump(plane_params, f, indent=4)

    print(f"The floor parameters are saved in {output_json}")


subprocess.run(["Exe_Helpers\mkv2ply.exe", VIDEO_PATH])
fit_plane_linear_regression("point_cloud_1_filtered.ply", "floor_params.json")
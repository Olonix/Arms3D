import matplotlib.pyplot as plt


def visualize_3d_as_plt(skel_3d, output_path):
    """
    Visualization of 3D skeleton using matplotlib library
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    points = {}
    valid_points_list = []
    for key, pt in skel_3d.items():
        if pt is not None:
            points[key] = pt
            valid_points_list.append(pt)

    if not points:
        print("No points to visualize")
        plt.close()
        return

    # Drawing points
    for key, pt in points.items():
        ax.scatter(pt[0], pt[1], pt[2], s=40, color='red')
        # Point labels
        # ax.text(pt[0] + 0.02, pt[1] + 0.02, pt[2], key, fontsize=7)

    # Function for drawing lines
    def try_line(pt1_key, pt2_key):
        pt1 = points.get(pt1_key)
        pt2 = points.get(pt2_key)
        if pt1 is not None and pt2 is not None:
            xs = [pt1[0], pt2[0]]
            ys = [pt1[1], pt2[1]]
            zs = [pt1[2], pt2[2]]
            ax.plot(xs, ys, zs, 'k-', linewidth=1.5)

    # Defining joints for a complete skeleton
    # Head and Torso
    try_line("head", "left_shoulder")
    try_line("head", "right_shoulder")
    try_line("left_shoulder", "right_shoulder")
    try_line("left_shoulder", "left_hip")
    try_line("right_shoulder", "right_hip")
    try_line("left_hip", "right_hip")
    # Left Arm
    try_line("left_shoulder", "left_elbow")
    try_line("left_elbow", "left_wrist")
    # Right Arm
    try_line("right_shoulder", "right_elbow")
    try_line("right_elbow", "right_wrist")
    # Left Leg
    try_line("left_hip", "left_knee")
    try_line("left_knee", "left_ankle")
    # Right Leg
    try_line("right_hip", "right_knee")
    try_line("right_knee", "right_ankle")

    # view settings
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.view_init(elev=50., azim=-30)

    # for debug
    # plt.show()

    if output_path:
        plt.savefig(output_path + ".jpg")

    plt.close()


# import plotly.graph_objects as go
# import numpy as np

# EDGES = [
#     ("head","left_shoulder"),("head","right_shoulder"),
#     ("left_shoulder","right_shoulder"),("left_shoulder","left_hip"),
#     ("right_shoulder","right_hip"),("left_hip","right_hip"),
#     ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
#     ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
#     ("left_hip","left_knee"),("left_knee","left_ankle"),
#     ("right_hip","right_knee"),("right_knee","right_ankle"),
# ]

# class PlotlyRenderer:
#     def __init__(self, width=800, height=800):
#         self.fig = go.Figure()
#         self.fig.add_trace(go.Scatter3d(
#             x=[], y=[], z=[],
#             mode='markers',
#             marker=dict(size=4, color='red')
#         ))
#         for _ in EDGES:
#             self.fig.add_trace(go.Scatter3d(
#                 x=[0,0], y=[0,0], z=[0,0],
#                 mode='lines',
#                 line=dict(width=2, color='black')
#             ))
#         self.fig.update_layout(
#             scene=dict(
#                 xaxis=dict(visible=False, range=[-1,1]),
#                 yaxis=dict(visible=False, range=[-1,1]),
#                 zaxis=dict(visible=False, range=[-1,1]),
#                 aspectmode='cube'
#             ),
#             margin=dict(l=0,r=0,t=0,b=0),
#             width=width, height=height
#         )

#     def update(self, skel_3d: dict):
#         keys, pts = [], []
#         for k,v in skel_3d.items():
#             if v is not None:
#                 keys.append(k); pts.append(v)
#         pts = np.array(pts)
#         idx = {k:i for i,k in enumerate(keys)}

#         self.fig.data[0].x = pts[:,0]
#         self.fig.data[0].y = pts[:,1]
#         self.fig.data[0].z = pts[:,2]

#         for i, (a, b) in enumerate(EDGES, start=1):
#             if a in idx and b in idx:
#                 p0 = pts[idx[a]]
#                 p1 = pts[idx[b]]
#                 self.fig.data[i].x = [p0[0], p1[0]]
#                 self.fig.data[i].y = [p0[1], p1[1]]
#                 self.fig.data[i].z = [p0[2], p1[2]]
#             else:
#                 self.fig.data[i].x = [0,0]
#                 self.fig.data[i].y = [0,0]
#                 self.fig.data[i].z = [0,0]

#     def render(self, output_path: str):
#         self.fig.write_image(output_path + ".png")
#         print(f"Saved {output_path}.png")
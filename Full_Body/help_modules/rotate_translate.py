import numpy as np


def load_plane_params(json_path):
    import json
    with open(json_path, 'r') as f:
        params = json.load(f)
    # plane: A x + B y + C z + D = 0
    n = np.array([params['A'], params['B'], params['C']], dtype=float)
    n /= np.linalg.norm(n)
    return n


def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.allclose(v, 0) and c > 0.9999:
        return np.eye(3)
    if np.allclose(v, 0) and c < -0.9999:
        # 180 degree rotation around any orthogonal axis
        # find orthogonal vector
        axis = np.array([1, 0, 0])
        if abs(a[0]) > 0.9:
            axis = np.array([0, 1, 0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        H = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
        return -np.eye(3) + 2 * np.outer(v, v)
    s = np.linalg.norm(v)
    kmat = np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],    0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return R


def rotate_skeleton(skeleton_3d: dict, R: np.ndarray) -> dict:
    """
    Apply rotation R to all 3D points in the skeleton dict (key -> [x,y,z] or None)
    """
    rotated = {}
    for key, pt in skeleton_3d.items():
        if pt is None:
            rotated[key] = None
        else:
            p = np.array(pt, dtype=float)
            rotated_p = R.dot(p)
            rotated[key] = rotated_p.tolist()
    return rotated

import numpy as np

def translate_skeleton_to_origin(skeleton, origin_point):
    """
    Moves the skeleton so that "origin_point" is at the origin
    """
    return {
        k: np.array(v) - origin_point
        for k, v in skeleton.items()
        if v is not None
    }

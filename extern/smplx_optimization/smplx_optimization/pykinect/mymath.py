import numpy as np
import torch.nn.functional as F
import torch


def rotate_a_b_axis_angle_torch_batched(a, b):
    a = a / torch.norm(a, dim=1, keepdim=True)
    b = b / torch.norm(b, dim=1, keepdim=True)
    rot_axis = torch.cross(a, b)

    a_proj = b * torch.sum(a * b, dim=1, keepdim=True)
    a_ort = a - a_proj
    theta = np.arctan2(
        torch.norm(a_ort, dim=1, keepdim=True),
        torch.norm(a_proj, dim=1, keepdim=True)
    )

    theta[torch.sum(a * b, dim=1) < 0] = np.pi - theta[torch.sum(a * b, dim=1) < 0]

    aa = rot_axis / torch.norm(rot_axis, dim=1, keepdim=True) * theta

    return aa


def rotate_a_b_axis_angle(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    rot_axis = np.cross(a, b)
#   find a proj onto b
    a_proj = b * (a.dot(b))
    a_ort = a - a_proj
#   find angle between a and b in [0, np.pi)
    theta = np.arctan2(np.linalg.norm(a_ort), np.linalg.norm(a_proj))
    if a.dot(b)<0:
        theta = np.pi - theta
    # print(theta)
    # print(rot_axis/np.linalg.norm(rot_axis))
    aa = rot_axis/np.linalg.norm(rot_axis)*theta
    # R, jac = cv2.Rodrigues(rot_axis)
    return aa


def align_canonical(x_dir, y_dir):
    x_dir /= np.linalg.norm(x_dir)
    z_dir = np.cross(x_dir, y_dir)
    z_dir /= np.linalg.norm(z_dir)
    y_dir = np.cross(z_dir, x_dir)
    y_dir /= np.linalg.norm(y_dir)
    R = np.stack([x_dir, y_dir, z_dir], axis=0)
    return R


def orthoprocrustes(xs, ys):
    n = xs.shape[0]
    xsm = np.mean(xs, 0)
    ysm = np.mean(ys, 0)
    q = xs.shape[1]
    dxs = xs - np.tile(xsm.reshape(1, -1), (n, 1))
    dys = ys - np.tile(ysm.reshape(1, -1), (n, 1))
    dxst = np.tile(dxs.reshape(-1, 1, q), (1, q, 1))
    dyst = np.tile(dys.reshape(-1, q, 1), (1, 1, q))
    S = np.sum(dyst * dxst, 0)
    u, s, v = np.linalg.svd(S)
    # print(u @ np.diag(s) @ v - S)
    s = np.sign(s)
    s = np.diag(s)
    R = u @ s @ v
    # print('ortoproc output')
    # print(R)
    # print(np.linalg.det(R))
    cnt = 0
    while (np.linalg.det(R) < 0):
        s[2 - cnt] *= -1
        R = u @ s @ v

    t = ysm - R @ xsm
    # print(np.linalg.norm(dxs @ R.T - dys))
    # print(np.linalg.norm(dys @ R.T - dxs))
    # n = xs.shape[0]
    # print(np.linalg.norm(xs @ R.T + np.tile(t.reshape(1, -1), (n, 1)) - ys))
    return R, t


def cam_project(points, K):
    """
    :param points: torch.Tensor of shape [b, n, 3]
    :param K: torch.Tensor intrinsics matrix of shape [b, 3, 3]
    :return: torch.Tensor points projected to 2d using K, shape: [b, n, 2]
    """
    b = points.shape[0]
    n = points.shape[1]

    points_K = torch.matmul(
        K.reshape(b, 1, 3, 3).repeat(1, n, 1, 1),
        points.reshape(b, n, 3, 1)
    )  # shape: [b, n, 3, 1]

    points_2d = points_K[:, :, :2, 0] / points_K[:, :, [2], 0]  # shape: [b, n, 2]

    return points_2d

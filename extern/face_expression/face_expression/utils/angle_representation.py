import torch
from torch.nn import functional as F
import numpy as np
import kornia
# from numba import jit, njit, prange


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def check_rep(rep):
    assert rep in ['aa', 'rotmtx', 'quat', 'rot6d']


def rep2shape(rep):
    check_rep(rep)
    if rep == 'aa':
        return list([3])
    elif rep == 'rotmtx':
        return list([3, 3])
    elif rep == 'quat':
        return list([4])
    elif rep == 'rot6d':
        return list([6])
    else:
        raise Exception()


def convert_impl(d, src_rep, dst_rep):
    """
    :param d: torch.Tensor of shape (N, *rep2shape(src_rep))
    :param src_rep: source representation
    :param dst_rep: destination representation
    :return: torch.Tensor of shape (N, *rep2shape(dst_rep))
    """
    check_rep(src_rep)
    check_rep(dst_rep)

    if src_rep == dst_rep:
        return d

    if dst_rep == 'rot6d':
        raise NotImplementedError

    if src_rep == 'rot6d':
        d = rot6d_to_rotmat(d)
        return convert_impl(d, 'rotmtx', dst_rep)
    else:
        rep_pair2f = {
            ('aa', 'rotmtx'): kornia.angle_axis_to_rotation_matrix,
            ('aa', 'quat'): kornia.angle_axis_to_quaternion,
            ('rotmtx', 'aa'): kornia.rotation_matrix_to_angle_axis,
            ('rotmtx', 'quat'): kornia.rotation_matrix_to_quaternion,
            ('quat', 'aa'): kornia.quaternion_to_angle_axis,
            ('quat', 'rotmtx'): kornia.quaternion_to_rotation_matrix
        }

        f = rep_pair2f[(src_rep, dst_rep)]

        if src_rep == 'aa' and dst_rep == 'quat':
            result_wxyz = f(d)
            result_xyzw = result_wxyz[:, [1, 2, 3, 0]]
            return result_xyzw
        elif src_rep == 'quat' and dst_rep == 'aa':
            d_xyzw = d
            d_wxyz = d_xyzw[:, [3, 0, 1, 2]]
            return f(d_wxyz)
        elif src_rep == 'rotmtx' and dst_rep == 'aa':
            return convert_impl(convert_impl(d, 'rotmtx', 'quat'), 'quat', 'aa')
        else:
            return f(d)


def rep_shape_forward(orig_shape, src_rep, dst_rep, flatten=False):
    cache = orig_shape, src_rep, dst_rep, flatten
    shape = [-1] + rep2shape(src_rep)
    return shape, cache


def rep_reshape_forward(d, src_rep, dst_rep, flatten=False):
    check_rep(src_rep)
    check_rep(dst_rep)
    orig_shape = d.shape
    shape, cache = rep_shape_forward(orig_shape, src_rep, dst_rep, flatten=flatten)
    return d.reshape(*shape), cache


def rep_shape_backward(cache):
    orig_shape, src_rep, dst_rep, flatten = cache
    src_rep_shape = rep2shape(src_rep)
    dst_rep_shape = rep2shape(dst_rep)

    if flatten:
        non_equal_shape_index = 0
    else:
        non_equal_shape_index = -1
        for i in range(len(src_rep_shape)):
            if orig_shape[-i - 1] != src_rep_shape[-i - 1]:
                non_equal_shape_index = i
                break

    if non_equal_shape_index == -1:
        dst_shape = list(orig_shape[:-len(src_rep_shape)]) + dst_rep_shape
    else:
        assert non_equal_shape_index == 0
        assert orig_shape[-1] % np.prod(src_rep_shape) == 0

        count = orig_shape[-1] / np.prod(src_rep_shape)
        dst_shape = list(orig_shape[:-1]) + [int(count * np.prod(dst_rep_shape))]
    return dst_shape


def rep_reshape(shape, src_rep, dst_rep, flatten=False):
    shape, cache = rep_shape_forward(shape, src_rep, dst_rep, flatten=flatten)
    dst_shape = rep_shape_backward(cache)
    return dst_shape


def rep_reshape_backward(d, cache):
    dst_shape = rep_shape_backward(cache)
    d = d.reshape(*dst_shape)
    return d


def convert(d, src_rep, dst_rep, flatten=False):
    check_rep(src_rep)
    check_rep(dst_rep)
    if src_rep == dst_rep:
        return d
    d, cache = rep_reshape_forward(d, src_rep, dst_rep, flatten=flatten)
    d = convert_impl(d, src_rep, dst_rep)
    d = rep_reshape_backward(d, cache)
    return d


def universe_convert(d, src_rep, dst_rep, flatten=False):
    check_rep(src_rep)
    check_rep(dst_rep)
    if src_rep == dst_rep:
        return d

    is_np = isinstance(d, np.ndarray)
    if is_np:
        d = torch.tensor(d, dtype=torch.float32, requires_grad=False)

    d = convert(d, src_rep, dst_rep, flatten=flatten)

    if is_np:
        d = d.numpy()
    return d


def is_valid_rotmat(rotmats, thresh=1e-6):
    """
    Checks that the rotation matrices are valid, i.e. R*R' == I and det(R) == 1
    Args:
        rotmats: A np array of shape (..., 3, 3).
        thresh: Numerical threshold.

    Returns:
        True if all rotation matrices are valid, False if at least one is not valid.
    """
    # check we have a valid rotation matrix
    rotmats_t = np.transpose(rotmats, tuple(range(len(rotmats.shape[:-2]))) + (-1, -2))
    mm = np.matmul(rotmats, rotmats_t)
    is_orthogonal = np.all(np.abs(mm - eye(3, rotmats.shape[:-2])) < thresh)
    det_is_one = np.all(np.abs(np.linalg.det(rotmats) - 1.0) < thresh)
    return is_orthogonal and det_is_one


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).

    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest


def rotmat2euler(rotmats):
    """
    Converts rotation matrices to euler angles. This is an adaptation of Martinez et al.'s code to work with batched
    inputs. Original code can be found here:
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/data_utils.py#L12

    Args:
        rotmats: An np array of shape (..., 3, 3)

    Returns:
        An np array of shape (..., 3) containing the Euler angles for each rotation matrix in `rotmats`
    """
    assert rotmats.shape[-1] == 3 and rotmats.shape[-2] == 3
    orig_shape = rotmats.shape[:-2]
    rs = np.reshape(rotmats, [-1, 3, 3])
    n_samples = rs.shape[0]

    # initialize to zeros
    e1 = np.zeros([n_samples])
    e2 = np.zeros([n_samples])
    e3 = np.zeros([n_samples])

    # find indices where we need to treat special cases
    is_one = rs[:, 0, 2] == 1
    is_minus_one = rs[:, 0, 2] == -1
    is_special = np.logical_or(is_one, is_minus_one)

    e1[is_special] = np.arctan2(rs[is_special, 0, 1], rs[is_special, 0, 2])
    e2[is_minus_one] = np.pi/2
    e2[is_one] = -np.pi/2

    # normal cases
    is_normal = ~np.logical_or(is_one, is_minus_one)
    # clip inputs to arcsin
    in_ = np.clip(rs[is_normal, 0, 2], -1, 1)
    e2[is_normal] = -np.arcsin(in_)
    e2_cos = np.cos(e2[is_normal])
    e1[is_normal] = np.arctan2(rs[is_normal, 1, 2]/e2_cos,
                               rs[is_normal, 2, 2]/e2_cos)
    e3[is_normal] = np.arctan2(rs[is_normal, 0, 1]/e2_cos,
                               rs[is_normal, 0, 0]/e2_cos)

    eul = np.stack([e1, e2, e3], axis=-1)
    eul = np.reshape(eul, np.concatenate([orig_shape, eul.shape[1:]]))
    return eul


def local_rot_to_global(joint_angles, parents, rep="rotmat", left_mult=False):
    """
    Converts local rotations into global rotations by "unrolling" the kinematic chain.
    Args:
        joint_angles: An np array of rotation matrices of shape (N, nr_joints*dof)
        parents: A np array specifying the parent for each joint
        rep: Which representation is used for `joint_angles`
        left_mult: If True the local matrix is multiplied from the left, rather than the right

    Returns:
        The global rotations as an np array of rotation matrices in format (N, nr_joints, 3, 3)
    """
    assert rep in ["rotmat", "quat", "aa"]
    n_joints = len(parents)
    if rep == "rotmat":
        rots = np.reshape(joint_angles, [-1, n_joints, 3, 3])
    elif rep == "quat":
        raise Exception()
        # rots = quaternion.as_rotation_matrix(quaternion.from_float_array(
        #     np.reshape(joint_angles, [-1, n_joints, 4])))
    else:
        raise Exception()
        # rots = quaternion.as_rotation_matrix(quaternion.from_rotation_vector(
        #     np.reshape(joint_angles, [-1, n_joints, 3])))

    out = np.zeros_like(rots)
    dof = rots.shape[-3]
    for j in range(dof):
        if parents[j] < 0:
            # root rotation
            out[..., j, :, :] = rots[..., j, :, :]
        else:
            parent_rot = out[..., parents[j], :, :]
            local_rot = rots[..., j, :, :]
            lm = local_rot if left_mult else parent_rot
            rm = parent_rot if left_mult else local_rot
            out[..., j, :, :] = np.matmul(lm, rm)
    return out


# @jit(fastmath=False, parallel=True, nopython=False)
def global_rot_to_local(rots, parents, left_mult=False):
    parents = np.array(parents)
    n_joints = parents.shape[0]
    orig_shape = rots.shape
    rots = rots.reshape((-1, n_joints, 3, 3))

    out = np.zeros_like(rots)
    for j in range(n_joints):
    # for j in prange(n_joints):
        if parents[j] < 0:
            out[:, j, :, :] = rots[:, j, :, :]
        else:
            parent_rot = rots[:, parents[j], :, :]
            parent_rot = np.transpose(parent_rot, (0, 2, 1))

            local_rot = rots[:, j, :, :]

            lm = local_rot if left_mult else parent_rot
            rm = parent_rot if left_mult else local_rot
            out[:, j, :, :] = np.matmul(lm, rm)
    out = out.reshape(*orig_shape)
    return out

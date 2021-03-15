# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import cv2
import time
# from numba import jit
# import numba

import torch
import torch.nn.functional as F

from .utils import rot_mat_to_euler

def find_dynamic_lmk_idx_and_bcoords_projective(vertices, neck_joint_loc, pose, dynamic_lmk_faces_idx,
                                     dynamic_lmk_b_coords,
                                     neck_kin_chain, dtype=torch.float32):
    batch_size = vertices.shape[0]
    # print('pose shape len')
    # print(len(pose.shape))
    if len(pose.shape) == 2:
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)
    else:
        rot_mats = torch.index_select(pose.view(batch_size, -1, 3, 3), 1,
                                     neck_kin_chain)

    b = rot_mats.shape[0]

    rel_rot_mat = torch.eye(3, device=vertices.device,
                            dtype=dtype).reshape(1, 3, 3).repeat(b, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
    neck_cam_loc = -torch.bmm(torch.transpose(rel_rot_mat, 1, 2), neck_joint_loc.reshape(-1, 3, 1)).reshape(-1, 3)
    neck_cam_loc_len = torch.norm(neck_cam_loc, dim=1).reshape(-1, 1).repeat(1, 3)
    neck_cam_dir = neck_cam_loc / neck_cam_loc_len
    y_ang = -torch.atan2(neck_cam_dir[:, 0], neck_cam_dir[:, 2])

    # print('neck cam dir')
    # print(neck_cam_dir)

    y_rot_angle = torch.round(
        torch.clamp(y_ang * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)

    # print('face cam ang')
    # print(y_rot_angle)

    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals +
                   (1 - neg_mask) * y_rot_angle)
    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                           0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                          0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def find_dynamic_lmk_idx_and_bcoords(vertices, pose, dynamic_lmk_faces_idx,
                                     dynamic_lmk_b_coords,
                                     neck_kin_chain, dtype=torch.float32):
    ''' Compute the faces, barycentric coordinates for the dynamic landmarks


        To do so, we first compute the rotation of the neck around the y-axis
        and then use a pre-computed look-up table to find the faces and the
        barycentric coordinates that will be used.

        Special thanks to Soubhik Sanyal (soubhik.sanyal@tuebingen.mpg.de)
        for providing the original TensorFlow implementation and for the LUT.

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        pose: torch.tensor Bx(Jx3), dtype = torch.float32
            The current pose of the body model
        dynamic_lmk_faces_idx: torch.tensor L, dtype = torch.long
            The look-up table from neck rotation to faces
        dynamic_lmk_b_coords: torch.tensor Lx3, dtype = torch.float32
            The look-up table from neck rotation to barycentric coordinates
        neck_kin_chain: list
            A python list that contains the indices of the joints that form the
            kinematic chain of the neck.
        dtype: torch.dtype, optional

        Returns
        -------
        dyn_lmk_faces_idx: torch.tensor, dtype = torch.long
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
        dyn_lmk_b_coords: torch.tensor, dtype = torch.float32
            A tensor of size BxL that contains the indices of the faces that
            will be used to compute the current dynamic landmarks.
    '''

    batch_size = vertices.shape[0]

    if len(pose.shape) == 2:
        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)
    else:
        rot_mats = torch.index_select(pose.view(batch_size, -1, 3, 3), 1,
                                      neck_kin_chain)

    b = rot_mats.shape[0]

    rel_rot_mat = torch.eye(3, device=vertices.device,
                            dtype=dtype).reshape(1, 3, 3).repeat(b, 1, 1)
    for idx in range(len(neck_kin_chain)):
        rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

    y_rot_angle = torch.round(
        torch.clamp(-rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                    max=39)).to(dtype=torch.long)
    neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
    mask = y_rot_angle.lt(-39).to(dtype=torch.long)
    neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
    y_rot_angle = (neg_mask * neg_vals +
                   (1 - neg_mask) * y_rot_angle)
#     print('std yrot ang')
#     print(y_rot_angle)

    dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                           0, y_rot_angle)
    dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                          0, y_rot_angle)

    return dyn_lmk_faces_idx, dyn_lmk_b_coords


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary_coords):
    ''' Calculates landmarks by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        faces: torch.tensor Fx3, dtype = torch.long
            The faces of the mesh
        lmk_faces_idx: torch.tensor L, dtype = torch.long
            The tensor with the indices of the faces used to calculate the
            landmarks.
        lmk_bary_coords: torch.tensor Lx3, dtype = torch.float32
            The tensor of barycentric coordinates that are used to interpolate
            the landmarks

        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
    '''
    # Extract the indices of the vertices for each face
    # BxLx3
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
        batch_size, -1, 3)

    lmk_faces += torch.arange(
        batch_size, dtype=torch.long, device=device).view(-1, 1, 1) * num_verts

    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3)

    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
    return landmarks


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # print('j1t')
    # print(J[0][1])

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location

    # print(rot_mats[0][1])

    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A, J


def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])

def vertices2joints_np(J_regressor_np, vertices_np):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''
    n_j, n_v = J_regressor_np.shape
    b = vertices_np.shape[0]
    return np.matmul(np.tile(J_regressor_np.reshape(1, n_j, n_v), (b, 1, 1)), vertices_np)


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape

def blend_shapes_np(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    # blend_shape = betas, shape_disps
    b = betas.shape[0]
    m, k, l = shape_disps.shape
    shape_disps_r = np.tile(shape_disps.reshape(1, m, k, l), (b, 1, 1, 1))
    betas_r = np.tile(betas.reshape(b, 1, -1, 1), (1, m, 1, 1))
    blend_shape = np.matmul(shape_disps_r, betas_r)

    return blend_shape.reshape((b, m, 3))

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def rodnumba(p):
    rot, jac = cv2.Rodrigues(p)
    return rot, jac


def batch_rodrigues_np(pose_body):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: ndarray Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: ndarray Nx3x3
            The rotation matrices for the given axis-angle parameters
        R_jac: ndarray Nx3x3x3
            Jacobians of the rotation matrices
    '''
    batch_size = pose_body.shape[0]
    n_j = int(pose_body.shape[1]/3)
    dt = pose_body.dtype
    rot_mats = np.zeros((batch_size, n_j, 3, 3), dtype=dt)
    rot_mats_jac = np.zeros((batch_size, n_j, 3, 3, 3), dtype=dt)
    for b in range(0, pose_body.shape[0]):
        for i in range(0, n_j):
            # rot, jac = cv2.Rodrigues(pose_body[b][3 * (i): 3 * (i + 1)].reshape(-1))
            rot, jac = rodnumba(pose_body[b][3 * (i): 3 * (i + 1)].reshape(-1))
            # print(numba.typeof(rot))
            # print(numba.typeof(jac))
            rot_mats[0, i] = rot
            rot_mats_jac[0, i] = jac.reshape(3, 3, 3)

    return rot_mats, rot_mats_jac

def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)

def transform_mat_np(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    b = R.shape[0]
    T = np.zeros((b, 4, 4), dtype=R.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t.reshape(b, 3)
    T[:, 3, 3] = 1
    return T


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    # print(transforms_mat[0][0])
    # print(transforms_mat[0][1])

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # print(transforms[0][1])

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms

# @jit
def batch_rigid_transform_diff(rot_mats, rot_mats_jac, transform_jac_chain, joints, parents, is_jac=True):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : ndarray BxNx3x3
        Tensor of rotation matrices
    rot_mats_jac : ndarray BxNx3x3x3
        Tensor of rotation matrix Jacobians
    joints : ndarray BxNx3
        Locations of joints
    parents : ndarray BxN
        The kinematic tree of each object

    Returns
    -------
    posed_joints : ndarray BxNx3
        The locations of the joints after applying the pose rotations
    pose_joints_jac : ndarray BxNxNx3x3
        Jacobian of pose_joints w.r.t. pose
    rel_transforms : ndarray BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    rel_transforms_jac : jacobian w.r.t. pose
    """
    # joints = joints.reshape()
    # joints = torch.unsqueeze(joints, dim=-1)

    t_0 = time.time()

    rel_joints = joints.copy()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat_np(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    b = joints.shape[0]
    n_j = joints.shape[1]

    t_1 = time.time()

    if is_jac:
        ainds = np.arange(0, parents.shape[0])
        transform_jac_chain.fill(0)
        transform_jac_chain[:, ainds, ainds, :, 0:3, 0:3] = rot_mats_jac[:, ainds]

    transform_chain = np.zeros((b, parents.shape[0], 4, 4), dtype=rot_mats.dtype)
    transform_chain[:, 0] = transforms_mat[:, 0]
    for i in range(1, parents.shape[0]):
        transform_chain[:, i] = np.matmul(transform_chain[:, parents[i]],
                                transforms_mat[:, i])

    if is_jac:
        inds = np.arange(1, len(parents))
        trans_parent = transform_chain[:, parents[inds]].reshape(b, len(inds), 1, 1, 4, 4)
        trans_jac = transform_jac_chain[:, inds, inds, :].reshape(1, len(inds), 1, 3, 4, 4)
        transform_jac_chain[:, inds, inds, :] = np.matmul(trans_parent, trans_jac).reshape(len(inds), 3, 4, 4)

        m = np.eye(parents.shape[0], dtype=np.bool)
        b = transforms_mat.shape[0]
        for i in range(1, parents.shape[0]):
            m[i] = m[i] | m[parents[i]]
            # transform_jac_chain[:, i, :, :] += np.matmul(transform_jac_chain[:, parents[i], :, :], transforms_mat[:, i])
            tr_jac_ch_sel = transform_jac_chain[:, parents[i], m[i], :]
            transform_jac_chain[:, i, m[i], :] += np.matmul(tr_jac_ch_sel, transforms_mat[:, i]).reshape(b, -1, 3, 4, 4)

    t_2 = time.time()


    transforms = transform_chain
    posed_joints = transforms[:, :, :3, 3]
    joints_rot = np.matmul(transforms[:, :, 0:3, 0:3], joints.reshape(b, n_j, 3, 1)).reshape((b, n_j, 3))
    rel_transforms = transforms.copy()
    rel_transforms[:, :, 0:3, 3] = rel_transforms[:, :, 0:3, 3] - joints_rot
    if is_jac:
        transforms_jac = np.transpose(transform_jac_chain, (0, 2, 3, 1, 4, 5))
        posed_joints_jac = transforms_jac[:, :, :, :, :3, 3]
        tjhj = np.matmul(transforms_jac[:, :, :, :, 0:3, 0:3], joints.reshape((b, 1, 1, n_j, 3, 1))).reshape(
            (b, n_j, 3, n_j, 3))
        rel_transforms_jac = transforms_jac.copy()
        rel_transforms_jac[:, :, :, :, 0:3, 3] = rel_transforms_jac[:, :, :, :, 0:3, 3] - tjhj
    else:
        posed_joints_jac = None
        rel_transforms_jac = None
    t_3 = time.time()

    # print('brgd breakdown {} {} {} '.format(t_1-t_0, t_2-t_1, t_3-t_2))

    return posed_joints, posed_joints_jac, rel_transforms, rel_transforms_jac

def batch_rigid_transform_fast_diff(rot_mats, rot_mats_jac, joints, parents):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : ndarray BxNx3x3
        Tensor of rotation matrices
    rot_mats_jac : ndarray BxNx3x3x3
        Tensor of rotation matrix Jacobians
    joints : ndarray BxNx3
        Locations of joints
    parents : ndarray BxN
        The kinematic tree of each object

    Returns
    -------
    posed_joints : ndarray BxNx3
        The locations of the joints after applying the pose rotations
    pose_joints_jac : ndarray BxNxNx3x3
        Jacobian of pose_joints w.r.t. pose
    rel_transforms : ndarray BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    rel_transforms_jac : jacobian w.r.t. pose
    """
    # joints = joints.reshape()
    # joints = torch.unsqueeze(joints, dim=-1)

    t_0 = time.time()

    rel_joints = joints.copy()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat_np(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    # print(transforms_mat[0][0])
    # print(transforms_mat[0][1])

    b = joints.shape[0]
    n_j = joints.shape[1]

    t_1 = time.time()

    # transform_jac = np.zeros((b, n_j, 3, 4, 4))
    # transform_jac[:, 0, :, 0:3, 0:3] = rot_mats_jac[:, 0]
    transform_jac_chain = np.zeros((b, n_j, 3, parents.shape[0], 4, 4))
    for i in range(0, parents.shape[0]):
        transform_jac_chain[:, i, :, i, 0:3, 0:3] = rot_mats_jac[:, i]

    transform_chain = np.copy(transforms_mat)
    for i in range(1, parents.shape[0]):
        t_curr = np.matmul(transform_chain[:, parents[i], 0:3, 0:3],
                                transforms_mat[:, i, 0:3, 3].reshape(-1, 1, 3, 1)).reshape(-1, 1, 3)
        transform_chain[:, i, 0:3, 3] = t_curr + transform_chain[:, parents[i], 0:3, 3]
        t_i = np.tile(transforms_mat[:, i, 0:3, 3].reshape(-1, 1, 1, 3, 1), (1, n_j, 3, 1, 1))
        # print(transform_jac_chain[:, :, :, parents[i], 0:3, 0:3].shape)
        # print(t_i.shape)
        t_jac_curr = np.matmul(transform_jac_chain[:, :, :, parents[i], 0:3, 0:3], t_i)
        transform_jac_chain[:, :, :, i, 0:3, 3] = transform_jac_chain[:, :, :, parents[i], 0:3, 3] + \
                                                  t_jac_curr.reshape(-1, n_j, 3, 3)

    # transforms = np.stack(transform_chain, axis=1)
    transforms = transform_chain
    # transforms_jac = np.stack(transform_jac_chain, axis=3)
    transforms_jac = transform_jac_chain

    t_2 = time.time()
    # print(transforms[0][1])

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]
    posed_joints_jac = transforms_jac[:, :, :, :, :3, 3]

    # joints_homogen = np.zeros((b, n_j, 4, 1))
    # joints_homogen[:, :, 0:3, 0] = joints

    joints_rot = np.matmul(transforms[:,:,0:3,0:3], joints.reshape(b, n_j, 3, 1)).reshape((b, n_j, 3))
    # jht = np.pad(np.matmul(transforms, joints_homogen), [(0, 0), (0, 0), (0, 0), (3,0)])
    # rel_transforms = transforms - jht
    rel_transforms = transforms.copy()
    rel_transforms[:, :, 0:3, 3] = rel_transforms[:, :, 0:3, 3] - joints_rot

    # jhtj = np.matmul(transforms_jac, joints_homogen.reshape((b, 1, 1, n_j, 4, 1)))
    # jhtj = np.pad(jhtj, [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (3, 0)])

    tjhj = np.matmul(transforms_jac[:,:,:,:,0:3,0:3], joints.reshape((b, 1, 1, n_j, 3, 1))).reshape((b, n_j, 3, n_j, 3))

    rel_transforms_jac = transforms_jac.copy()
    rel_transforms_jac[:, :, :, :, 0:3, 3] = rel_transforms_jac[:, :, :, :, 0:3, 3] - tjhj
    # rel_transforms_jac = transforms_jac - jhtj

    t_3 = time.time()

    print('brgd breakdown {} {} {} '.format(t_1-t_0, t_2-t_1, t_3-t_2))

    return posed_joints, posed_joints_jac, rel_transforms, rel_transforms_jac

def prepare_J(betas, v_template, shapedirs, J_regressor, n_v):
    # Add shape contribution
    v_shaped = v_template + blend_shapes_np(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints_np(J_regressor, v_shaped)

    n_j = J_regressor.shape[0]

    batch_size = betas.shape[0]

    homogen_coord = np.ones((batch_size, n_v, 1), dtype=v_template.dtype)

    transform_jac_chain = np.zeros((batch_size, n_j, n_j, 3, 4, 4), dtype=v_template.dtype)

    return J, v_shaped, homogen_coord, transform_jac_chain

def rel_to_direct(pose, parents):

    rot_mats, rot_mat_jacs = batch_rodrigues_np(pose)

    b = pose.shape[0]

    rot_chain = np.zeros((b, parents.shape[0], 3, 3))
    rot_chain[:, 0] = rot_mats[:, 0]
    for i in range(1, parents.shape[0]):
        rot_chain[:, i] = np.matmul(rot_chain[:, parents[i]], rot_mats[:, i])

    pose_dir = np.zeros_like(pose)
    n_j = int(pose.shape[1]/3)
    for b in range(0, pose.shape[0]):
        for i in range(0, n_j):
            rv, jac = cv2.Rodrigues(rot_chain[b, i])
            pose_dir[b, 3*i : 3*(i+1)] = rv.reshape(-1)

    return pose_dir



def lbs_diff_fast(pose, parents,
        J, v_shaped, W, W_j, homogen_coord, v_inds=None):
    t_0 = time.time()

    batch_size = pose.shape[0]

    # print('j1:')
    # print(J[0][1])

    # 3. Add pose blend shapes
    # N x J x 3 x 3

    t_1 = time.time()

    rot_mats, rot_mat_jacs = batch_rodrigues_np(pose)
    # rot_mats = rot_mats.reshape((batch_size, -1, 3, 3))
    n_j = rot_mats.shape[1]

    if v_inds is not None:
        v_shaped = v_shaped[:, v_inds, :]

    n_v = v_shaped.shape[1]

    v_posed = v_shaped

    t_2 = time.time()

    J_transformed, J_transformed_jac, A, A_jac = batch_rigid_transform_fast_diff(rot_mats, rot_mat_jacs, J, parents)

    t_3 = time.time()

    t_3 = time.time()

    # 5. Do skinning:
    # W is N x V x (J + 1)
    # W = np.tile(lbs_weights.reshape(1, n_v, n_j), (batch_size, 1, 1))
    # (N x V x (J + 1)) x (N x (J + 1) x 16) = N x V x 16
    num_joints = n_j
    T = np.matmul(W, A.reshape(batch_size, num_joints, 16)) \
        .reshape((batch_size, -1, 4, 4))

    # W_j = np.tile(W.reshape((batch_size, 1, 1, n_v, n_j)), (1, n_j, 3, 1, 1))

    A_jact = A_jac  # .transpose(0, 2, 3, 1, 4, 5)
    T_jac = np.matmul(W_j, A_jact.reshape(batch_size, n_j, 3, n_j, -1))
    T_jac = T_jac.reshape((batch_size, n_j, 3, n_v, 4, 4))
    # N x n_j x 3 x V x 16

    v_posed_homo = np.concatenate([v_posed, homogen_coord], axis=2)

    v_homo = np.matmul(T, v_posed_homo.reshape((batch_size, n_v, 4, 1)))

    T_j = T.reshape((batch_size, 1, 1, n_v, 4, 4))

    v_posed_homo_j = v_posed_homo.reshape((batch_size, 1, 1, n_v, 4, 1))
    v_homo_jac2 = np.matmul(T_jac, v_posed_homo_j)

    verts = v_homo[:, :, :3, 0]

    v_homo_jac = v_homo_jac2[:, :, :, :, :3, :]

    verts_jac = v_homo_jac[:, :, :, :, :3, 0]

    t_4 = time.time()

    # print('breakdown b {} a {} rt {} f {}'.format(t_1-t_0, t_2-t_1, t_3-t_2, t_4-t_3))

    return verts, verts_jac, J_transformed, J_transformed_jac, A, A_jac, J  # , v_posed, v_posed_jac

# @jit
def lbs_diff(pose, posedirs, parents, J, v_shaped, lbs_weights, homogen_coord, transform_jac_chain, v_inds=None):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : ndarray Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template ndarray BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : ndarray Px(V * 3)
            The pose PCA coefficients
        J_regressor : ndarray JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: ndarray J
            The array that describes the kinematic tree for the model
        lbs_weights: ndarray N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        vinds: ndarray - list of required vertex indices (if None, all vertices will be processed)

        Returns
        -------
        verts: ndarray BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        verts_jac: ndarray BxVx(J+1)x3x3
        joints: ndarray BxJx3
            The joints of the model
        joints_jac: ndarray BxJxJx3x3
            Jacobian of joints' coordinates
    '''

    t_0 = time.time()

    batch_size = pose.shape[0]

    if v_inds is not None:
        lbs_weights = lbs_weights[v_inds]
        n_v = len(v_inds)
    else:
        n_v = 0

    n_j = J.shape[1]

    W = np.tile(lbs_weights.reshape(1, n_v, n_j), (batch_size, 1, 1))
    # W_j = np.tile(W.reshape((batch_size, 1, 1, n_v, n_j)), (1, n_j, 3, 1, 1))
    W_j = W.reshape((batch_size, 1, 1, n_v, n_j))

    # print('j1:')
    # print(J[0][1])

    # 3. Add pose blend shapes
    # N x J x 3 x 3

    t_1 = time.time()

    rot_mats, rot_mat_jacs = batch_rodrigues_np(pose)
    # rot_mats = rot_mats.reshape((batch_size, -1, 3, 3))
    n_j = rot_mats.shape[1]
    pose_feature = (rot_mats[:, 1:, :, :] - np.tile(np.eye(3).reshape(1, 1, 3, 3), (batch_size, n_j-1, 1, 1)))\
        .reshape((batch_size, -1))

    if v_inds is not None:
        v_shaped = v_shaped[:, v_inds, :]
        inds = np.stack([3*v_inds, 3*v_inds+1, 3*v_inds+2], axis=1).reshape(-1)
        posedirs = posedirs[:, inds]
        # lbs_weights = lbs_weights[v_inds]

        # (N x P) x (P, V * 3) -> N x V x 3
    pose_offsets = np.matmul(pose_feature, posedirs) \
        .reshape((batch_size, -1, 3))

    n_v = v_shaped.shape[1]

    pose_offset_jacs = np.zeros((batch_size, n_j, 3, n_v, 3), dtype=pose.dtype)
    for i in range (1, n_j):
        pdi = np.matmul(rot_mat_jacs[:, i].reshape(-1, 9), posedirs[(i-1)*9 : i*9, :]).reshape((batch_size, 3, -1, 3))
        pose_offset_jacs[:, i] = pdi

    v_posed = pose_offsets + v_shaped

    #NxVxJ+1x3x3
    v_posed_jac = pose_offset_jacs

    # print(rot_mats[0][1])

    t_2 = time.time()

    J_transformed, J_transformed_jac, A, A_jac = batch_rigid_transform_diff(rot_mats, rot_mat_jacs, transform_jac_chain, J, parents)

    t_3 = time.time()

    # 5. Do skinning:
    # W is N x V x (J + 1)
    # W = np.tile(lbs_weights.reshape(1, n_v, n_j), (batch_size, 1, 1))
    # (N x V x (J + 1)) x (N x (J + 1) x 16) = N x V x 16
    num_joints = n_j
    T = np.matmul(W, A.reshape(batch_size, num_joints, 16)) \
        .reshape((batch_size, -1, 4, 4))

    # W_j = np.tile(W.reshape((batch_size, 1, 1, n_v, n_j)), (1, n_j, 3, 1, 1))

    A_jact = A_jac #.transpose(0, 2, 3, 1, 4, 5)
    T_jac = np.matmul(W_j, A_jact.reshape(batch_size, n_j, 3, n_j, -1))
    T_jac = T_jac.reshape((batch_size, n_j, 3, n_v, 4, 4))
    #N x n_j x 3 x V x 16

    v_posed_homo = np.concatenate([v_posed, homogen_coord], axis=2)

    v_homo = np.matmul(T, v_posed_homo.reshape((batch_size, n_v, 4, 1)))

    # T_j = np.tile(T.reshape((batch_size, 1, 1, n_v, 4, 4)), (1, n_j, 3, 1, 1, 1))

    T_j = T.reshape((batch_size, 1, 1, n_v, 4, 4))

    # v_posed_jac_h = np.pad(v_posed_jac, ((0, 0), (0,0), (0,0), (0,0), (0,1))).reshape((batch_size, n_j, 3, n_v, 4, 1))
    # v_posed_jac_h = v_posed_jac_h.reshape((batch_size, n_j, 3, n_v, 4, 1))
    # v_homo_jac1 = np.matmul(T_j, v_posed_jac_h)

    v_homo_jac1 = np.matmul(T_j[:, :, :, :, 0:3, 0:3], v_posed_jac.reshape((batch_size, n_j, 3, n_v, 3, 1)))

    # v_posed_homo_j = np.tile(v_posed_homo.reshape((batch_size, 1, 1, n_v, 4, 1)), (1, n_j, 3, 1, 1, 1))
    v_posed_homo_j = v_posed_homo.reshape((batch_size, 1, 1, n_v, 4, 1))
    v_homo_jac2 = np.matmul(T_jac, v_posed_homo_j)
    # v_homo_jac2 = v_homo_jac2.transpose((0, 3, 1, 2, 4, 5))

    verts = v_homo[:, :, :3, 0]

    v_homo_jac = v_homo_jac1 + v_homo_jac2[:, :, :, :,  :3, :]

    verts_jac = v_homo_jac[:, :, :, :, :3, 0]

    t_4 = time.time()

    # print('breakdown b {} a {} rt {} f {}'.format(t_1-t_0, t_2-t_1, t_3-t_2, t_4-t_3))

    return verts, verts_jac, J_transformed, J_transformed_jac, A, A_jac, J #, v_posed, v_posed_jac


def lbs_diff_nopd(pose, posedirs_face, parents, J, v_shaped, lbs_weights, homogen_coord, transform_jac_chain, v_inds=None,
                  is_jac=True, bpm=None, face_expression=None, w_thr=0):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : ndarray Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template ndarray BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : ndarray Px(V * 3)
            The pose PCA coefficients
        J_regressor : ndarray JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: ndarray J
            The array that describes the kinematic tree for the model
        lbs_weights: ndarray N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        vinds: ndarray - list of required vertex indices (if None, all vertices will be processed)

        Returns
        -------
        verts: ndarray BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        verts_jac: ndarray BxVx(J+1)x3x3
        joints: ndarray BxJx3
            The joints of the model
        joints_jac: ndarray BxJxJx3x3
            Jacobian of joints' coordinates
    '''

    t_0 = time.time()

    batch_size = pose.shape[0]
    if v_inds is not None:

        if bpm is None:
            w_arr = np.sum(lbs_weights[v_inds] ** 2, axis=0)/len(v_inds)
            bpm = w_arr > w_thr

        lbs_weights = lbs_weights[v_inds]

    # 3. Add pose blend shapes
    # N x J x 3 x 3

    t_1 = time.time()

    rot_mats, rot_mat_jacs = batch_rodrigues_np(pose)

    n_j = rot_mats.shape[1]

    if v_inds is not None:
        v_shaped = v_shaped[:, v_inds, :]

        if face_expression is not None:
            posedirs_face = posedirs_face[v_inds]
            n_v = len(v_inds)
            dv = np.matmul(posedirs_face.reshape(1, n_v, 3, 1, -1), np.tile(face_expression.reshape(1, 1, 1, -1, 1),
                                                                            (1, n_v, 3, 1, 1)))
            v_shaped += dv.reshape(1, -1, 3)
            dv_jac = np.transpose(posedirs_face.reshape(n_v, 3, -1), (2, 0, 1)).reshape(-1, 3 * n_v)

    n_v = v_shaped.shape[1]

    v_posed = v_shaped

    t_2 = time.time()

    J_transformed, J_transformed_jac, A, A_jac = batch_rigid_transform_diff(rot_mats, rot_mat_jacs, transform_jac_chain, J, parents, is_jac)

    t_3 = time.time()

    # 5. Do skinning:
    # W is N x V x (J + 1)
    # W = np.tile(lbs_weights.reshape(1, n_v, n_j), (batch_size, 1, 1))
    # (N x V x (J + 1)) x (N x (J + 1) x 16) = N x V x 16
    num_joints = n_j
    W = np.tile(lbs_weights.reshape(1, n_v, n_j), (batch_size, 1, 1))
    T = np.matmul(W, A.reshape(batch_size, num_joints, 16)) \
        .reshape((batch_size, -1, 4, 4))
    v_posed_homo = np.concatenate([v_posed, homogen_coord], axis=2)
    v_homo = np.matmul(T, v_posed_homo.reshape((batch_size, n_v, 4, 1)))
    verts = v_homo[:, :, :3, 0]

    t_3_1 = time.time()

    if is_jac:

        Wred = W[:, :, bpm]
        n_bpm = Wred.shape[-1]
        W_j = Wred.reshape((batch_size, 1, 1, n_v, n_bpm))
        T_jac = np.matmul(W_j, A_jac.reshape(batch_size, n_j, 3, n_j, -1)[:, :, :, bpm])
        T_jac = T_jac.reshape((batch_size, n_j, 3, n_v, 4, 4))

        #N x n_j x 3 x V x 16

        t_3_2 = time.time()

        v_posed_homo_j = v_posed_homo.reshape((batch_size, 1, 1, n_v, 4, 1))
        v_homo_jac2 = np.matmul(T_jac, v_posed_homo_j)
        n_pars = pose.shape[1]
        verts_jac = v_homo_jac2[:, :, :, :,  :3, 0].reshape(n_pars, -1)
        if face_expression is not None:
            n_fp = 10
            dv_jac = np.matmul(T[:, :, 0:3, 0:3], dv_jac.reshape(n_fp, -1, 3, 1)).reshape(n_fp, -1)
            # dv_jac = np.zeros_like(dv_jac)
            verts_jac = np.concatenate([verts_jac, dv_jac], axis=0)
    else:
        verts_jac = None

    t_4 = time.time()

    # print('breakdown b {} a {} rt {} f1 {} f2 {} f3 {}'.format(t_1-t_0, t_2-t_1, t_3-t_2, t_3_1 - t_3, t_3_2 - t_3_1, t_4-t_3_2))

    return verts, verts_jac, J_transformed, J_transformed_jac, A, A_jac, J, bpm #, v_posed, v_posed_jac
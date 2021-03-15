import numpy as np
import cv2
import torch
import time

from smplx.lbs import batch_rigid_transform_diff, batch_rodrigues_np, \
    lbs_diff, prepare_J, lbs_diff_fast, rel_to_direct, lbs_diff_nopd
from smplx.body_models import SMPLX, SMPL, SMPLH
import trimesh

def fun1(batch_size, n_j, pose_body, J, parents):
    #batch rodrigues
    rot_mats, rot_mat_jacs = batch_rodrigues_np(pose_body)

    # 4. Get the global joint location
    transform_jac_chain = np.zeros((batch_size, n_j, n_j, 3, 4, 4))
    return batch_rigid_transform_diff(rot_mats, rot_mat_jacs, transform_jac_chain, J, parents)


def test_batch_rigid_transform_diff(data_path):
    batch_size = 1
    bm_path = data_path + '/body_models/smplx/SMPLX_MALE.npz'
    smpl_dict = np.load(bm_path, encoding='latin1', allow_pickle=True)

    p2n = smpl_dict['part2num']
    w = smpl_dict['weights']
    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    njoints = smpl_dict['posedirs'].shape[2] // 3
    model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano'}[njoints]

    v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)

    num_total_betas = smpl_dict['shapedirs'].shape[-1]
    if num_betas < 1:
        num_betas = num_total_betas

    shapedirs = smpl_dict['shapedirs'][:, :, :]

    # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
    posedirs = smpl_dict['posedirs']
    posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
    # indices of parents for each joints
    kintree_table = smpl_dict['kintree_table'].astype(np.int32)
    parents = kintree_table[0]
    # LBS weights
    # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
    weights = smpl_dict['weights']

    J_regressor = smpl_dict['J_regressor']

    trans = np.zeros((batch_size, 3))

    # root_orient
    # if self.model_type in ['smpl', 'smplh']:
    root_orient = np.zeros((batch_size, 3))

    # pose_body
    pose_body = np.zeros((batch_size, 63))

    n_j_b = 21

    # pose_hand

    pose_hand = np.zeros((batch_size, 1 * 3 * 2))

    betas = np.zeros((batch_size, num_betas))

    expression = np.zeros((batch_size, num_betas))

    shape_components = np.concatenate([betas, expression], axis=1)

    # Add shape contribution
    v_shaped = v_template

    # Get the joints
    # NxJx3 array

    n_j = SMPLX.NUM_JOINTS + 1
    J = np.zeros((batch_size, n_j, 3))
    # J = vertices2joints(J_regressor, v_shaped)
    J[0] = J_regressor @ v_shaped

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    pose = np.zeros((batch_size, n_j * 3))
    pose[0, 0:3] = np.asarray([-2.98, 0.07, -0.08])

    # test tensor vs ndarray
    betas_t = torch.tensor(betas, dtype=torch.float)
    global_orient = torch.tensor(pose[:, :3], dtype=torch.float)
    body_pose = torch.tensor(pose[:, 3:3 * (SMPLX.NUM_BODY_JOINTS + 1)], dtype=torch.float)
    lh_pose = torch.tensor(
        pose[:, 3 * (SMPLX.NUM_BODY_JOINTS + 1): 3 * (SMPLX.NUM_BODY_JOINTS + SMPLX.NUM_HAND_JOINTS + 1)],
        dtype=torch.float)
    rh_pose = torch.tensor(
        pose[:, 3 * (SMPLX.NUM_BODY_JOINTS + SMPLX.NUM_HAND_JOINTS + 1): 3 * (
                    SMPLX.NUM_BODY_JOINTS + 2 * SMPLX.NUM_HAND_JOINTS + 1)],
        dtype=torch.float)
    smpl = SMPLX(model_path=bm_path, ext='npz', betas=betas_t, use_pca=False, flat_hand_mean=True,
                 use_face_contour=True)
    bm = smpl(global_orient=global_orient, body_pose=body_pose, left_hand_pose=lh_pose, right_hand_pose=rh_pose)

    verts = bm.vertices[0].detach().cpu().numpy()
    faces = smpl.faces

    v_t = bm.vertices.detach().cpu().numpy()
    j_t = bm.joints.detach().cpu().numpy()

    v_inds = np.arange(0, 100)
    n_v = len(v_inds)

    J, v_shaped, homogen_coord, transform_jac_chain = prepare_J(shape_components, v_template, shapedirs, J_regressor,
                                                                n_v)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    pose = np.zeros((batch_size, n_j*3))
    J_transformed, J_transformec_jac, A, A_jac = fun1(batch_size, n_j, pose, J, parents)

    delta = 1e-8
    thr_acc = 1e-5
    for rot_id in range(0, n_j):
        for i in range(0, 3):
            pose_p = pose.copy()
            pose_p[0][rot_id * 3 + i] += delta
            J_transformed_p, J_transformec_jac_p, A_p, A_jac_p = fun1(batch_size, n_j, pose_p, J, parents)
            dJ = 1.0/delta * (J_transformed_p - J_transformed)
            dJ_pred = J_transformec_jac[:, rot_id, i]
            dA = 1.0/delta * (A_p - A)
            dA_pred = A_jac[:, rot_id, i]
            j_err = np.linalg.norm(dJ - dJ_pred)
            a_err = np.linalg.norm(dA - dA_pred)
            assert (a_err < thr_acc)
            assert(j_err < thr_acc)

    print('test finished')


def test_lbs_diff(data_path):
    batch_size = 1
    bm_path = data_path + '/body_models/smplx/SMPLX_MALE.npz'
    smpl_dict = np.load(bm_path, encoding='latin1', allow_pickle=True)

    p2n = smpl_dict['part2num']
    w = smpl_dict['weights']
    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    njoints = smpl_dict['posedirs'].shape[2] // 3
    model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano'}[njoints]

    v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)

    num_total_betas = smpl_dict['shapedirs'].shape[-1]
    if num_betas < 1:
        num_betas = num_total_betas

    shapedirs = smpl_dict['shapedirs'][:, :, :]

    # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
    posedirs = smpl_dict['posedirs']
    posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
    # indices of parents for each joints
    kintree_table = smpl_dict['kintree_table'].astype(np.int32)
    parents = kintree_table[0]
    # LBS weights
    # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
    weights = smpl_dict['weights']


    J_regressor = smpl_dict['J_regressor']

    trans = np.zeros((batch_size, 3))

    # root_orient
    # if self.model_type in ['smpl', 'smplh']:
    root_orient = np.zeros((batch_size, 3))

    # pose_body
    pose_body = np.zeros((batch_size, 63))

    n_j_b = 21

    # pose_hand

    pose_hand = np.zeros((batch_size, 1 * 3 * 2))

    betas = np.zeros((batch_size, num_betas))

    expression = np.zeros((batch_size, num_betas))

    shape_components = np.concatenate([betas, expression], axis=1)

    # Add shape contribution
    v_shaped = v_template

    # Get the joints
    # NxJx3 array

    n_j = SMPLX.NUM_JOINTS + 1
    J = np.zeros((batch_size, n_j, 3))
    #J = vertices2joints(J_regressor, v_shaped)
    J[0] = J_regressor @ v_shaped

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    pose = np.zeros((batch_size, n_j*3))
    pose[0, 0:3] = np.asarray([-2.98, 0.07, -0.08])

    #test tensor vs ndarray
    betas_t = torch.tensor(betas, dtype=torch.float)
    global_orient = torch.tensor(pose[:, :3], dtype=torch.float)
    body_pose = torch.tensor(pose[:, 3:3*(SMPLX.NUM_BODY_JOINTS+1)], dtype=torch.float)
    lh_pose = torch.tensor(pose[:, 3*(SMPLX.NUM_BODY_JOINTS+1) : 3*(SMPLX.NUM_BODY_JOINTS+SMPLX.NUM_HAND_JOINTS+1)], dtype=torch.float)
    rh_pose = torch.tensor(
        pose[:, 3 * (SMPLX.NUM_BODY_JOINTS + SMPLX.NUM_HAND_JOINTS + 1) : 3 * (SMPLX.NUM_BODY_JOINTS + 2*SMPLX.NUM_HAND_JOINTS + 1)],
        dtype=torch.float)
    smpl = SMPLX(model_path=bm_path, ext='npz', betas=betas_t, use_pca=False, flat_hand_mean=True, use_face_contour=True)
    bm = smpl(global_orient=global_orient, body_pose=body_pose, left_hand_pose=lh_pose, right_hand_pose=rh_pose)


    verts = bm.vertices[0].detach().cpu().numpy()
    faces = smpl.faces

    curr_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    obj_txt = trimesh.exchange.obj.export_obj(curr_mesh)
    with open('smplx.obj', 'w') as f_out:
        f_out.write(obj_txt)


    v_t = bm.vertices.detach().cpu().numpy()
    j_t = bm.joints.detach().cpu().numpy()

    v_inds = np.arange(0, 100)
    n_v = len(v_inds)

    J, v_shaped, homogen_coord, transform_jac_chain = prepare_J(shape_components, v_template, shapedirs, J_regressor, n_v)


    verts, verts_jac, J_transformed, J_transformed_jac, A, A_jac, J = lbs_diff(pose,
                                                                               posedirs,
                                                                               parents,
                                                                               J, v_shaped,
                                                                               weights,
                                                                               homogen_coord,
                                                                               transform_jac_chain,
                                                                               v_inds)

    j_err = np.linalg.norm(j_t[:, 0:n_j] - J_transformed)
    v_err = np.linalg.norm(v_t[:, v_inds] - verts)
    print('nd-t errors joint {} vert {}'.format(j_err, v_err))
    # if j_err > 1e-5:
    #     for i in range(0, n_j):
    #         print(i)
    #         print(j_t[0][i])
    #         print(J_transformed[0][i])
    v_id = 99
    delta = 1e-8
    thr_acc = 1e-5

    for rot_id in range(0, n_j):
        for i in range(0, 3):
            pose_p = pose.copy()
            pose_p[0][rot_id * 3 + i] += delta
            # J_transformed_p, J_transformec_jac_p, A_p, A_jac_p = fun1(batch_size, n_j, pose_p, J, parents)
            t_0 = time.time()
            verts_p, verts_jac_p, J_transformed_p, J_transformed_jac_p, A_p, A_jac_p, J_p = \
                lbs_diff(pose_p, posedirs, parents, J, v_shaped, weights, homogen_coord, transform_jac_chain, v_inds)
            t_1 = time.time()
            # print('whole time {} s '.format(t_1-t_0))
            dJ = 1.0/delta * (J_transformed_p - J_transformed)
            dJ_pred = J_transformed_jac[:, rot_id, i]
            dA = 1.0/delta * (A_p - A)
            dA_pred = A_jac[:, rot_id, i]
            dV = 1.0/delta * (verts_p - verts)
            dV_pred = verts_jac[:, rot_id, i]
            v_err = np.linalg.norm(dV_pred[0, v_id] - dV[0, v_id])
            # print(v_err)
            assert (v_err < thr_acc)

            # if v_err > thr_acc:
            #     print(dV_pred[0, v_id])
            #     print(dV[0, v_id])
            #     print('-')

            jac_err = np.linalg.norm(dJ - dJ_pred)
            print('{} {} : {} {}'.format(rot_id, i, jac_err, v_err))
            assert (jac_err < thr_acc)

            # if jac_err > thr_acc:
            #     print('---')
            #     for ii in range(0, 55):
            #         print(dJ[0,ii])
            #         print(dJ_pred[0,ii])
            #         print('-')
            #
            # print('-')

            a_err = np.linalg.norm(dA - dA_pred)

            assert (a_err < thr_acc)

            # if a_err > thr_acc:
            #     for ii in range(0, 55):
            #         print(dJ[0,ii])
            #         print(dJ_pred[0,ii])
            #         print('-')

    print('test finished')


def test_lbs_diff_nopd():
    batch_size = 1
    data_path = '/home/alexander/projects/pykinect_prev/data/'
    # data_path = '/storage/projects/pykinect/data/'
    bm_path = data_path + '/body_models/smplx/SMPLX_MALE.npz'
    smpl_dict = np.load(bm_path, encoding='latin1', allow_pickle=True)

    p2n = smpl_dict['part2num']
    w = smpl_dict['weights']
    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    njoints = smpl_dict['posedirs'].shape[2] // 3
    model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano'}[njoints]

    v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)

    num_total_betas = smpl_dict['shapedirs'].shape[-1]
    if num_betas < 1:
        num_betas = num_total_betas

    shapedirs = smpl_dict['shapedirs'][:, :, :]

    # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
    posedirs = smpl_dict['posedirs']
    posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
    # indices of parents for each joints
    kintree_table = smpl_dict['kintree_table'].astype(np.int32)
    parents = kintree_table[0]
    # LBS weights
    # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
    weights = smpl_dict['weights']


    J_regressor = smpl_dict['J_regressor']

    trans = np.zeros((batch_size, 3))

    # root_orient
    # if self.model_type in ['smpl', 'smplh']:
    root_orient = np.zeros((batch_size, 3))

    # pose_body
    pose_body = np.zeros((batch_size, 63))

    n_j_b = 21

    # pose_hand

    pose_hand = np.zeros((batch_size, 1 * 3 * 2))

    betas = np.zeros((batch_size, num_betas))

    expression = np.zeros((batch_size, num_betas))

    shape_components = np.concatenate([betas, expression], axis=1)

    # Add shape contribution
    v_shaped = v_template

    # Get the joints
    # NxJx3 array

    n_j = SMPLX.NUM_JOINTS + 1
    J = np.zeros((batch_size, n_j, 3))
    #J = vertices2joints(J_regressor, v_shaped)
    J[0] = J_regressor @ v_shaped

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    pose = np.zeros((batch_size, n_j*3))
    pose[0, 0:3] = np.asarray([-2.98, 0.07, -0.08])

    #test tensor vs ndarray
    betas_t = torch.tensor(betas, dtype=torch.float)
    global_orient = torch.tensor(pose[:, :3], dtype=torch.float)
    body_pose = torch.tensor(pose[:, 3:3*(SMPLX.NUM_BODY_JOINTS+1)], dtype=torch.float)
    lh_pose = torch.tensor(pose[:, 3*(SMPLX.NUM_BODY_JOINTS+1) : 3*(SMPLX.NUM_BODY_JOINTS+SMPLX.NUM_HAND_JOINTS+1)], dtype=torch.float)
    rh_pose = torch.tensor(
        pose[:, 3 * (SMPLX.NUM_BODY_JOINTS + SMPLX.NUM_HAND_JOINTS + 1) : 3 * (SMPLX.NUM_BODY_JOINTS + 2*SMPLX.NUM_HAND_JOINTS + 1)],
        dtype=torch.float)
    smpl = SMPLX(model_path=bm_path, ext='npz', betas=betas_t, use_pca=False, flat_hand_mean=True, use_face_contour=True)
    bm = smpl(global_orient=global_orient, body_pose=body_pose, left_hand_pose=lh_pose, right_hand_pose=rh_pose)


    verts = bm.vertices[0].detach().cpu().numpy()
    faces = smpl.faces

    curr_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    obj_txt = trimesh.exchange.obj.export_obj(curr_mesh)
    with open('smplx.obj', 'w') as f_out:
        f_out.write(obj_txt)


    v_t = bm.vertices.detach().cpu().numpy()
    j_t = bm.joints.detach().cpu().numpy()

    v_inds = np.arange(0, 100)
    n_v = len(v_inds)

    J, v_shaped, homogen_coord, transform_jac_chain = prepare_J(shape_components, v_template, shapedirs, J_regressor, n_v)


    verts, verts_jac, J_transformed, J_transformed_jac, A, A_jac, J, bpm = lbs_diff_nopd(pose,
                                                                               posedirs,
                                                                               parents,
                                                                               J, v_shaped,
                                                                               weights,
                                                                               homogen_coord,
                                                                               transform_jac_chain,
                                                                               v_inds)

    j_err = np.linalg.norm(j_t[:, 0:n_j] - J_transformed)
    v_err = np.linalg.norm(v_t[:, v_inds] - verts)
    print('nd-t errors joint {} vert {}'.format(j_err, v_err))
    if j_err > 1e-5:
        for i in range(0, n_j):
            print(i)
            print(j_t[0][i])
            print(J_transformed[0][i])
    v_id = 99
    delta = 1e-6
    for rot_id in range(1, n_j):
        for i in range(0, 3):
            pose_p = pose.copy()
            pose_p[0][rot_id * 3 + i] += delta
            # J_transformed_p, J_transformec_jac_p, A_p, A_jac_p = fun1(batch_size, n_j, pose_p, J, parents)
            t_0 = time.time()
            verts_p, verts_jac_p, J_transformed_p, J_transformed_jac_p, A_p, A_jac_p, J_p, bpm = \
                lbs_diff_nopd(pose_p, posedirs, parents, J, v_shaped, weights, homogen_coord, transform_jac_chain, v_inds, bpm=bpm)
            t_1 = time.time()
            # print('whole time {} s '.format(t_1-t_0))
            dJ = 1.0/delta * (J_transformed_p - J_transformed)
            dJ_pred = J_transformed_jac[:, rot_id, i]
            dA = 1.0/delta * (A_p - A)
            dA_pred = A_jac[:, rot_id, i]
            dV = 1.0/delta * (verts_p - verts)
            dV_pred = verts_jac[:, rot_id, i]
            v_err = np.linalg.norm(dV_pred[0, v_id] - dV[0, v_id])
            # print(v_err)
            # if v_err > 1e-5:
            #     print(dV_pred[0, v_id])
            #     print(dV[0, v_id])

            # dT = 1.0/delta * (T_p - T)
            # dT_pred = T_jac[:, rot_id, i]
            # print(np.linalg.norm(dT - dT_pred))
            jac_err = np.linalg.norm(dJ - dJ_pred)
            print('{} {} : {} {}'.format(rot_id, i, jac_err, v_err))
            # if jac_err > 0.1:
            #     print('---')
            # for ii in range(0, 55):
            #     print(dJ[0,ii])
            #     print(dJ_pred[0,ii])
            #     print('-')

            print('-')
            # print(np.linalg.norm(dA - dA_pred))

def test_lbs_diff_nopd(data_path):
    batch_size = 1

    bm_path = data_path + '/body_models/smplx/SMPLX_MALE.npz'
    smpl_dict = np.load(bm_path, encoding='latin1', allow_pickle=True)

    p2n = smpl_dict['part2num']
    w = smpl_dict['weights']
    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    njoints = smpl_dict['posedirs'].shape[2] // 3

    v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)

    num_total_betas = smpl_dict['shapedirs'].shape[-1]
    if num_betas < 1:
        num_betas = num_total_betas

    shapedirs = smpl_dict['shapedirs'][:, :, :]

    shapedirs_face = shapedirs[:, :, -10:]

    # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
    posedirs = smpl_dict['posedirs']
    posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
    # indices of parents for each joints
    kintree_table = smpl_dict['kintree_table'].astype(np.int32)
    parents = kintree_table[0]
    # LBS weights
    # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
    weights = smpl_dict['weights']


    J_regressor = smpl_dict['J_regressor']

    trans = np.zeros((batch_size, 3))

    # root_orient
    # if self.model_type in ['smpl', 'smplh']:
    root_orient = np.zeros((batch_size, 3))

    # pose_body
    pose_body = np.zeros((batch_size, 63))

    n_j_b = 21

    # pose_hand

    pose_hand = np.zeros((batch_size, 1 * 3 * 2))

    betas = np.zeros((batch_size, num_betas))

    expression = np.zeros((batch_size, num_betas))

    shape_components = np.concatenate([betas, expression], axis=1)

    # Add shape contribution
    v_shaped = v_template

    # Get the joints
    # NxJx3 array

    n_j = SMPLX.NUM_JOINTS + 1
    J = np.zeros((batch_size, n_j, 3))
    #J = vertices2joints(J_regressor, v_shaped)
    J[0] = J_regressor @ v_shaped

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    pose = np.zeros((batch_size, n_j*3))
    pose[0, 0:3] = np.asarray([-2.98, 0.07, -0.08])

    #test tensor vs ndarray
    betas_t = torch.tensor(betas, dtype=torch.float)
    global_orient = torch.tensor(pose[:, :3], dtype=torch.float)
    body_pose = torch.tensor(pose[:, 3:3*(SMPLX.NUM_BODY_JOINTS+1)], dtype=torch.float)
    lh_pose = torch.tensor(pose[:, 3*(SMPLX.NUM_BODY_JOINTS+1) : 3*(SMPLX.NUM_BODY_JOINTS+SMPLX.NUM_HAND_JOINTS+1)], dtype=torch.float)
    rh_pose = torch.tensor(
        pose[:, 3 * (SMPLX.NUM_BODY_JOINTS + SMPLX.NUM_HAND_JOINTS + 1) : 3 * (SMPLX.NUM_BODY_JOINTS + 2*SMPLX.NUM_HAND_JOINTS + 1)],
        dtype=torch.float)
    smpl = SMPLX(model_path=bm_path, ext='npz', betas=betas_t, use_pca=False, flat_hand_mean=True, use_face_contour=True)
    bm = smpl(global_orient=global_orient, body_pose=body_pose, left_hand_pose=lh_pose, right_hand_pose=rh_pose)


    verts = bm.vertices[0].detach().cpu().numpy()
    faces = smpl.faces

    # curr_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # obj_txt = trimesh.exchange.obj.export_obj(curr_mesh)
    # with open('smplx.obj', 'w') as f_out:
    #     f_out.write(obj_txt)


    v_t = bm.vertices.detach().cpu().numpy()
    j_t = bm.joints.detach().cpu().numpy()

    v_inds = np.arange(0, 100)
    n_v = len(v_inds)

    shape_components[0, 0:10] = np.random.randn(10)

    J, v_shaped, homogen_coord, transform_jac_chain = prepare_J(shape_components, v_template, shapedirs, J_regressor, n_v)

    face_expr_0 = np.random.rand(10)

    verts, verts_jac, J_transformed, J_transformed_jac, A, A_jac, J, bpm = lbs_diff_nopd(pose,
                                                                               shapedirs_face,
                                                                               parents,
                                                                               J, v_shaped,
                                                                               weights,
                                                                               homogen_coord,
                                                                               transform_jac_chain,
                                                                               v_inds,
                                                                               face_expression=face_expr_0)

    verts_jac = verts_jac[:-10].reshape(1, n_j, 3, len(v_inds), 3)

    # verts_jac.fill(0)
    # verts_jac = verts_jac[-10:]
    j_err = np.linalg.norm(j_t[:, 0:n_j] - J_transformed)
    v_err = np.linalg.norm(v_t[:, v_inds] - verts)
    print('nd-t errors joint {} vert {}'.format(j_err, v_err))
    # if j_err > 1e-5:
    #     for i in range(0, n_j):
    #         print(i)
    #         print(j_t[0][i])
    #         print(J_transformed[0][i])
    v_id = 99
    delta = 1e-8
    thr_acc = 1e-5

    for rot_id in range(0, n_j):
        for i in range(0, 3):
            pose_p = pose.copy()
            pose_p[0][rot_id * 3 + i] += delta
            # J_transformed_p, J_transformec_jac_p, A_p, A_jac_p = fun1(batch_size, n_j, pose_p, J, parents)
            t_0 = time.time()
            verts_p, verts_jac_p, J_transformed_p, J_transformed_jac_p, A_p, A_jac_p, J_p, bpm = \
                lbs_diff_nopd(pose_p, shapedirs_face, parents, J, v_shaped, weights, homogen_coord, transform_jac_chain,
                              v_inds, bpm=bpm, face_expression=face_expr_0)
            t_1 = time.time()
            dJ = 1.0/delta * (J_transformed_p - J_transformed)
            dJ_pred = J_transformed_jac[:, rot_id, i]
            dA = 1.0/delta * (A_p - A)
            dA_pred = A_jac[:, rot_id, i]
            dV = 1.0/delta * (verts_p - verts)
            dV_pred = verts_jac[:, rot_id, i]
            v_err = np.linalg.norm(dV_pred[0, v_id] - dV[0, v_id])
            # print(v_err)
            # assert (v_err < thr_acc)

            # if v_err > thr_acc:
            #     print(dV_pred[0, v_id])
            #     print(dV[0, v_id])
            #     print('-')

            jac_err = np.linalg.norm(dJ - dJ_pred)
            print('{} {} : {} {}'.format(rot_id, i, jac_err, v_err))
            # assert (jac_err < thr_acc)

            if jac_err > thr_acc:
                print('---')
                for ii in range(0, 55):
                    print(dJ[0,ii])
                    print(dJ_pred[0,ii])
                    print('-')
            #
            print('-')

            a_err = np.linalg.norm(dA - dA_pred)

            assert (a_err < thr_acc)

            # if a_err > thr_acc:
            #     for ii in range(0, 55):
            #         print(dJ[0,ii])
            #         print(dJ_pred[0,ii])
            #         print('-')

            assert(v_err < 1e-5)

            if v_err > 1e-5:
                for v_id in range(0, dV.shape[1]):
                    if (np.linalg.norm(dV_pred.reshape(-1, 3)[v_id] - dV[0, v_id]) > 1e-3):
                        print(dV_pred.reshape(-1, 3)[v_id])
                        print(dV[0, v_id])
                        print('-')
            print('--')

            # dT = 1.0/delta * (T_p - T)
            # dT_pred = T_jac[:, rot_id, i]
            # print(np.linalg.norm(dT - dT_pred))
            # jac_err = np.linalg.norm(dJ - dJ_pred)
            # print('{} {} : {} {}'.format(rot_id, i, jac_err, v_err))
            # if jac_err > 0.1:
            #     print('---')
            # for ii in range(0, 55):
            #     print(dJ[0,ii])
            #     print(dJ_pred[0,ii])
            #     print('-')

            print('-')
        print('face expression diff test passed')

def test_lbs_diff_faceexpr_nopd(data_path):
    batch_size = 1

    bm_path = data_path + '/body_models/smplx/SMPLX_MALE.npz'
    smpl_dict = np.load(bm_path, encoding='latin1', allow_pickle=True)

    p2n = smpl_dict['part2num']
    w = smpl_dict['weights']
    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    njoints = smpl_dict['posedirs'].shape[2] // 3

    v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)

    num_total_betas = smpl_dict['shapedirs'].shape[-1]
    if num_betas < 1:
        num_betas = num_total_betas

    shapedirs = smpl_dict['shapedirs'][:, :, :]

    shapedirs_face = shapedirs[:, :, -10:]

    # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
    posedirs = smpl_dict['posedirs']
    posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
    # indices of parents for each joints
    kintree_table = smpl_dict['kintree_table'].astype(np.int32)
    parents = kintree_table[0]
    # LBS weights
    # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
    weights = smpl_dict['weights']


    J_regressor = smpl_dict['J_regressor']

    trans = np.zeros((batch_size, 3))

    # root_orient
    # if self.model_type in ['smpl', 'smplh']:
    root_orient = np.zeros((batch_size, 3))

    # pose_body
    pose_body = np.zeros((batch_size, 63))

    n_j_b = 21

    # pose_hand

    pose_hand = np.zeros((batch_size, 1 * 3 * 2))

    betas = np.zeros((batch_size, num_betas))

    expression = np.zeros((batch_size, num_betas))

    shape_components = np.concatenate([betas, expression], axis=1)

    # Add shape contribution
    v_shaped = v_template

    # Get the joints
    # NxJx3 array

    n_j = SMPLX.NUM_JOINTS + 1
    J = np.zeros((batch_size, n_j, 3))
    #J = vertices2joints(J_regressor, v_shaped)
    J[0] = J_regressor @ v_shaped

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    pose = np.zeros((batch_size, n_j*3))
    pose[0, 0:3] = np.asarray([-2.98, 0.07, -0.08])

    #test tensor vs ndarray
    betas_t = torch.tensor(betas, dtype=torch.float)
    global_orient = torch.tensor(pose[:, :3], dtype=torch.float)
    body_pose = torch.tensor(pose[:, 3:3*(SMPLX.NUM_BODY_JOINTS+1)], dtype=torch.float)
    lh_pose = torch.tensor(pose[:, 3*(SMPLX.NUM_BODY_JOINTS+1) : 3*(SMPLX.NUM_BODY_JOINTS+SMPLX.NUM_HAND_JOINTS+1)], dtype=torch.float)
    rh_pose = torch.tensor(
        pose[:, 3 * (SMPLX.NUM_BODY_JOINTS + SMPLX.NUM_HAND_JOINTS + 1) : 3 * (SMPLX.NUM_BODY_JOINTS + 2*SMPLX.NUM_HAND_JOINTS + 1)],
        dtype=torch.float)
    smpl = SMPLX(model_path=bm_path, ext='npz', betas=betas_t, use_pca=False, flat_hand_mean=True, use_face_contour=True)
    bm = smpl(global_orient=global_orient, body_pose=body_pose, left_hand_pose=lh_pose, right_hand_pose=rh_pose)


    verts = bm.vertices[0].detach().cpu().numpy()
    faces = smpl.faces

    # curr_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # obj_txt = trimesh.exchange.obj.export_obj(curr_mesh)
    # with open('smplx.obj', 'w') as f_out:
    #     f_out.write(obj_txt)


    v_t = bm.vertices.detach().cpu().numpy()
    j_t = bm.joints.detach().cpu().numpy()

    v_inds = np.arange(0, 10000)
    n_v = len(v_inds)

    shape_components[0, 0:10] = np.random.randn(10)

    J, v_shaped, homogen_coord, transform_jac_chain = prepare_J(shape_components, v_template, shapedirs, J_regressor, n_v)

    face_expr_0 = np.random.rand(10)

    verts, verts_jac, J_transformed, J_transformed_jac, A, A_jac, J, bpm = lbs_diff_nopd(pose,
                                                                               shapedirs_face,
                                                                               parents,
                                                                               J, v_shaped,
                                                                               weights,
                                                                               homogen_coord,
                                                                               transform_jac_chain,
                                                                               v_inds,
                                                                               face_expression=face_expr_0)

    # verts_jac.fill(0)
    verts_jac = verts_jac[-10:]
    j_err = np.linalg.norm(j_t[:, 0:n_j] - J_transformed)
    v_err = np.linalg.norm(v_t[:, v_inds] - verts)
    print('nd-t errors joint {} vert {}'.format(j_err, v_err))
    # if j_err > 1e-5:
    #     for i in range(0, n_j):
    #         print(i)
    #         print(j_t[0][i])
    #         print(J_transformed[0][i])
    v_id = 99
    delta = 1e-8

    for i in range(0, 10):
        fe_p = face_expr_0.copy()
        fe_p[i] += delta
        # J_transformed_p, J_transformec_jac_p, A_p, A_jac_p = fun1(batch_size, n_j, pose_p, J, parents)
        t_0 = time.time()
        verts_p, verts_jac_p, J_transformed_p, J_transformed_jac_p, A_p, A_jac_p, J_p, bpm = \
            lbs_diff_nopd(pose, shapedirs_face, parents, J, v_shaped, weights, homogen_coord, transform_jac_chain,
                          v_inds, bpm=bpm, face_expression=fe_p)
        t_1 = time.time()
        # print('whole time {} s '.format(t_1-t_0))
        dV = 1.0/delta * (verts_p - verts)
        dV_pred = verts_jac[i]
        v_err = np.linalg.norm(dV_pred.reshape(-1, 3) - dV[0])

        assert(v_err < 1e-5)

        if v_err > 1e-5:
            for v_id in range(0, dV.shape[1]):
                if (np.linalg.norm(dV_pred.reshape(-1, 3)[v_id] - dV[0, v_id]) > 1e-3):
                    print(dV_pred.reshape(-1, 3)[v_id])
                    print(dV[0, v_id])
                    print('-')
        print('--')

        # dT = 1.0/delta * (T_p - T)
        # dT_pred = T_jac[:, rot_id, i]
        # print(np.linalg.norm(dT - dT_pred))
        # jac_err = np.linalg.norm(dJ - dJ_pred)
        # print('{} {} : {} {}'.format(rot_id, i, jac_err, v_err))
        # if jac_err > 0.1:
        #     print('---')
        # for ii in range(0, 55):
        #     print(dJ[0,ii])
        #     print(dJ_pred[0,ii])
        #     print('-')

        print('-')
    print('face expression diff test passed')

def test_lbs_diff_fast():
    batch_size = 1
    data_path = '/home/alexander/projects/pykinect/data/'
    # data_path = '/storage/projects/pykinect/data/'
    bm_path = data_path + '/body_models/smplx/SMPLX_MALE.npz'
    smpl_dict = np.load(bm_path, encoding='latin1', allow_pickle=True)

    p2n = smpl_dict['part2num']
    w = smpl_dict['weights']
    num_betas = 10  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters

    njoints = smpl_dict['posedirs'].shape[2] // 3
    model_type = {69: 'smpl', 153: 'smplh', 162: 'smplx', 45: 'mano'}[njoints]

    v_template = np.repeat(smpl_dict['v_template'][np.newaxis], batch_size, axis=0)

    num_total_betas = smpl_dict['shapedirs'].shape[-1]
    if num_betas < 1:
        num_betas = num_total_betas

    shapedirs = smpl_dict['shapedirs'][:, :, :]

    # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
    posedirs = smpl_dict['posedirs']
    posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
    # indices of parents for each joints
    kintree_table = smpl_dict['kintree_table'].astype(np.int32)
    parents = kintree_table[0]
    # LBS weights
    # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
    weights = smpl_dict['weights']


    J_regressor = smpl_dict['J_regressor']

    trans = np.zeros((batch_size, 3))

    # root_orient
    # if self.model_type in ['smpl', 'smplh']:
    root_orient = np.zeros((batch_size, 3))

    # pose_body
    pose_body = np.zeros((batch_size, 63))

    n_j_b = 21

    # pose_hand

    pose_hand = np.zeros((batch_size, 1 * 3 * 2))

    betas = np.zeros((batch_size, num_betas))

    expression = np.zeros((batch_size, num_betas))

    shape_components = np.concatenate([betas, expression], axis=1)

    # Add shape contribution
    v_shaped = v_template

    # Get the joints
    # NxJx3 array

    n_j = SMPLX.NUM_JOINTS + 1
    J = np.zeros((batch_size, n_j, 3))
    #J = vertices2joints(J_regressor, v_shaped)
    J[0] = J_regressor @ v_shaped

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    pose = np.zeros((batch_size, n_j*3))
    pose[0, 0:3] = np.asarray([-2.98, 0.07, -0.08])

    #test tensor vs ndarray
    betas_t = torch.tensor(betas, dtype=torch.float)
    global_orient = torch.tensor(pose[:, :3], dtype=torch.float)
    body_pose = torch.tensor(pose[:, 3:3*(SMPLX.NUM_BODY_JOINTS+1)], dtype=torch.float)
    lh_pose = torch.tensor(pose[:, 3*(SMPLX.NUM_BODY_JOINTS+1) : 3*(SMPLX.NUM_BODY_JOINTS+SMPLX.NUM_HAND_JOINTS+1)], dtype=torch.float)
    rh_pose = torch.tensor(
        pose[:, 3 * (SMPLX.NUM_BODY_JOINTS + SMPLX.NUM_HAND_JOINTS + 1) : 3 * (SMPLX.NUM_BODY_JOINTS + 2*SMPLX.NUM_HAND_JOINTS + 1)],
        dtype=torch.float)
    smpl = SMPLX(model_path=bm_path, ext='npz', betas=betas_t, use_pca=False, flat_hand_mean=True, use_face_contour=True)
    bm = smpl(global_orient=global_orient, body_pose=body_pose, left_hand_pose=lh_pose, right_hand_pose=rh_pose)


    verts = bm.vertices[0].detach().cpu().numpy()


    v_t = bm.vertices.detach().cpu().numpy()
    j_t = bm.joints.detach().cpu().numpy()

    v_inds = np.arange(0, 100)

    J, v_shaped, W, W_j, homogen_coord = prepare_J(shape_components, v_template, shapedirs, J_regressor, weights, v_inds)

    pose_direct = rel_to_direct(pose, parents)
    verts, verts_jac, J_transformed, J_transformed_jac, A, A_jac, J = lbs_diff_fast(pose_direct,
                                                                                    parents,
                                                                                    J, v_shaped,
                                                                                    W, W_j,
                                                                                    homogen_coord,
                                                                                    v_inds)

    v_id = 99
    delta = 1e-6
    for rot_id in range(1, n_j):
        for i in range(0, 3):
            pose_p = pose_direct.copy()
            pose_p[0][rot_id * 3 + i] += delta
            # J_transformed_p, J_transformec_jac_p, A_p, A_jac_p = fun1(batch_size, n_j, pose_p, J, parents)
            t_0 = time.time()
            verts_p, verts_jac_p, J_transformed_p, J_transformed_jac_p, A_p, A_jac_p, J_p = \
                lbs_diff_fast(pose_p, parents, J, v_shaped, W, W_j, homogen_coord, v_inds)
            t_1 = time.time()
            # print('whole time {} s '.format(t_1-t_0))
            dJ = 1.0/delta * (J_transformed_p - J_transformed)
            dJ_pred = J_transformed_jac[:, rot_id, i]
            dA = 1.0/delta * (A_p - A)
            dA_pred = A_jac[:, rot_id, i]
            dV = 1.0/delta * (verts_p - verts)
            dV_pred = verts_jac[:, rot_id, i]
            v_err = np.linalg.norm(dV_pred[0, v_id] - dV[0, v_id])
            a_err = np.linalg.norm(dA_pred[:, rot_id, i] - dA[:, rot_id, i])
            # print(v_err)
            # if v_err > 1e-5:
            #     print(dV_pred[0, v_id])
            #     print(dV[0, v_id])

            # dT = 1.0/delta * (T_p - T)
            # dT_pred = T_jac[:, rot_id, i]
            # print(np.linalg.norm(dT - dT_pred))
            jac_err = np.linalg.norm(dJ - dJ_pred)
            print('{} {} : {} {} {}'.format(rot_id, i, jac_err, v_err, a_err))
            if jac_err > 0.1:
                print('---')
            # for ii in range(0, 55):
            #     print(dJ[0,ii])
            #     print(dJ_pred[0,ii])
            #     print('-')

            print('-')
            # print(np.linalg.norm(dA - dA_pred))


if __name__ == '__main__':
    data_path = '/home/alexander/projects/pykinect/data/'
    data_path = '/storage/projects/pykinect/data/'
    test_batch_rigid_transform_diff(data_path)
    test_lbs_diff(data_path)
    test_lbs_diff_nopd(data_path)
    test_lbs_diff_faceexpr_nopd(data_path)
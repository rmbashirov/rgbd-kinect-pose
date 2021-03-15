import sys
import inspect
import os.path as osp
import numpy as np
from copy import deepcopy
from scipy.special import softmax
import torch

from smplx_optimization.pykinect.pose_init import initialize_pose_advanced
from smplx_optimization.pykinect.smplx_model import ExpBodyModel

from patched_smplx.utils import to_np

from smplx_kinect.common.angle_representation import universe_convert


def to_t(d, device, dtype=torch.float32):
    return torch.tensor(d, device=device, dtype=dtype)


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


def get_smplx_init(
    kinect_joints, kinect_confs, betas,
    kintree_table, T, s2k, J
):
    betas = torch.tensor(betas)

    parents = to_np(kintree_table[0], dtype=int)
    parents[0] = -1

    kinect_confs = kinect_confs.reshape((32, 1))
    kinect_confs = np.repeat(kinect_confs, 3, axis=1)

    joints_kinect_d = kinect_joints
    joints_viz_f = kinect_confs

    joints_kinect_m = joints_kinect_d @ T[0:3, 0:3].T + T[0:3, 3].reshape(1, 3)

    dtype = np.float32
    v_kin_flat = s2k @ np.concatenate([betas.reshape(10), np.ones(1)])
    v_kin = v_kin_flat.reshape(-1, 3).astype(dtype)

    rots, trans = initialize_pose_advanced(joints_kinect_m, joints_viz_f[:, 0], v_kin, J, parents, dtype)
    if rots is None:
        print('Perform pose init failed, taking default values')
        rots = np.zeros((len(parents), 3), dtype=dtype)
        trans = np.zeros(3, dtype=dtype)

    rot = rots[0]
    pose_body = rots[1:22].reshape(-1)

    return pose_body, rot, trans


def load_exp_bm(pykinect_data_dp, gender, n_pca, device):
    if gender == 'male':
        bm_path = osp.join(pykinect_data_dp, 'body_models/smplx/SMPLX_MALE.npz')
        s2k_path = osp.join(pykinect_data_dp, 'rob75_val/s2k_m.npy')
    elif gender == 'female':
        bm_path = osp.join(pykinect_data_dp, 'body_models/smplx/SMPLX_FEMALE.npz')
        s2k_path = osp.join(pykinect_data_dp, 'rob75_val/s2k_f.npy')
    else:
        raise Exception(f'gender {gender} unknown')

    smpl_dict = np.load(bm_path, allow_pickle=True)
    kintree_table = smpl_dict['kintree_table']

    s2k = np.load(s2k_path)

    kinect_vert_weights_path = osp.join(pykinect_data_dp, 'rob75_val/weights.npy')
    w_add = np.load(kinect_vert_weights_path)
    w_add = softmax(w_add, axis=1)

    exp_bm = ExpBodyModel(
        bm_path,
        is_hand_pca=True,
        num_hand_pca=n_pca,
        fe_scale=10000,
        s2v=s2k,
        w_add=w_add,
        comp_device=device
    )

    J_path = osp.join(pykinect_data_dp, 'rob75_val/J.npy')
    J = np.load(J_path)

    return exp_bm, s2k, J, kintree_table


def inf_exp_bm(
    exp_bm, device,
    global_trans=None, global_rot=None, pose_body=None,
    face_expression=None, pose_jaw=None, pose_eyes=None,
    pose_hand=None, n_pca=None,
    beta=None, allow_beta_none=False):

    if beta is None:
        assert allow_beta_none

    def to_batch_t(x, size=None):
        if x is None:
            if size is None:
                raise Exception()
            x = np.zeros(size)
        return to_t(x, device=device).reshape(1, -1)

    global_trans = to_batch_t(global_trans, 3)
    global_rot = to_batch_t(global_rot, 3)
    pose_body = to_batch_t(pose_body, 63)
    face_expression = to_batch_t(face_expression, 10)
    pose_jaw = to_batch_t(pose_jaw, 3)
    pose_eyes = to_batch_t(pose_eyes, 6)
    pose_hand = to_batch_t(pose_hand, 2 * n_pca if n_pca is not None else None)
    beta = to_batch_t(beta, 10)

    exp_bm_out = exp_bm(
        global_rot, pose_body, pose_hand, pose_jaw, pose_eyes, beta, global_trans, face_expression)
    return exp_bm_out


def exp_bm_out2kinect_joints(exp_bm_out, substract_pelvis=False, return_smplx_pelvis=False):
    # verts = exp_bm_out.v[0].detach().cpu().numpy()
    # j3d_pred = verts[-32:]
    j3d_pred = exp_bm_out.v[0, -32:].detach().cpu().numpy()
    A = exp_bm_out.A

    if substract_pelvis:
        pelvis = deepcopy(j3d_pred[[0], :])
        j3d_pred -= pelvis

    if return_smplx_pelvis:
        smplx_pelvis = exp_bm_out.Jtr[0, 0].detach().cpu().numpy()
        return j3d_pred, A, smplx_pelvis
    else:
        return j3d_pred, A


def calc_kinect_twists(kinect_joints, init_kinect_joints, a_inv, bones):
    bones = torch.tensor(bones)

    kinect_bones = torch.from_numpy(
        kinect_joints[bones[:, 0]] - kinect_joints[bones[:, 1]]
    ).type(torch.float32)
    kinect_bones = torch.bmm(kinect_bones.unsqueeze(1), a_inv[bones[:, 0]]).squeeze(1)

    init_kinect_bone = torch.from_numpy(
        init_kinect_joints[bones[:, 0]] - init_kinect_joints[bones[:, 1]]
    ).type(torch.float32)
    init_kinect_bone = torch.bmm(init_kinect_bone.unsqueeze(1), a_inv[bones[:, 0]]).squeeze(1)

    kinect_twists = rotate_a_b_axis_angle_torch_batched(kinect_bones, init_kinect_bone)

    return kinect_twists.numpy()


class ExpBMWrapper:
    def __init__(self, pykinect_data_dp, device, n_pca=12, override_kinect2smplx_mapping=True):
        self.pykinect_data_dp = pykinect_data_dp
        self.device = device
        self.n_pca = n_pca

        kintree_dp = osp.join(self.pykinect_data_dp, 'kintree_kinect')
        self.kintree = np.loadtxt(kintree_dp, dtype=int)
        bones = []
        for child, parent in enumerate(self.kintree):
            bones.append([parent, child + 1])
        self.bones = np.array(bones)

        if override_kinect2smplx_mapping:
            self.bodyparts = np.array([
                0, 3, 6, 9, 13, 16, 18, 20, 20, 20, 20, 14, 17,
                19, 21, 21, 21, 21, 1, 4, 7, 10, 2, 5, 8,
                11, 12, 15, 15, 15, 15, 15])
        else:
            bp_weights_dp = osp.join(self.pykinect_data_dp, 'rob75_val/weights.npy')
            bp_weights = np.load(bp_weights_dp)
            self.bodyparts = np.argmax(bp_weights[:, :21], axis=1)

        self.pykinect_data = dict()
        for gender in ['female', 'male']:
            exp_bm, s2k, J, kintree_table = load_exp_bm(self.pykinect_data_dp, gender, n_pca, device)
            exp_bm.to(device)
            self.pykinect_data[gender] = {
                'exp_bm': exp_bm,
                's2k': s2k,
                'J': J,
                'kintree_table': kintree_table
            }

        self.T = np.eye(4)

    def get_smplx_init(self, kinect_joints_mm, kinect_confs, betas, gender):
        gender_data = self.pykinect_data[gender]

        pose_body, rot, trans = get_smplx_init(
            kinect_joints=kinect_joints_mm,
            kinect_confs=kinect_confs,
            betas=betas,

            kintree_table=gender_data['kintree_table'],
            T=self.T,
            s2k=gender_data['s2k'],
            J=gender_data['J']
        )

        return pose_body, rot, trans

    def inf_exp_bm(
        self,
        gender='male',
        global_trans=None, global_rot=None, pose_body=None,
        face_expression=None, pose_jaw=None, pose_eyes=None,
        pose_hand=None, n_pca=None,
        beta=None, allow_beta_none=False
    ):
        exp_bm_out = inf_exp_bm(
            exp_bm=self.pykinect_data[gender]['exp_bm'], device=self.device,
            global_trans=global_trans,
            global_rot=global_rot,
            pose_body=pose_body,
            face_expression=face_expression,
            pose_jaw=pose_jaw,
            pose_eyes=pose_eyes,
            pose_hand=pose_hand,
            n_pca=n_pca if n_pca is not None else self.n_pca,
            beta=beta,
            allow_beta_none=allow_beta_none
        )
        return exp_bm_out

    def get_twists_v2(self, init_A, init_kinect_joints, target_kinect_joints):
        init_A = init_A.clone().detach().cpu()[0, :, :3, :3]
        init_A_select = torch.index_select(init_A, 0, torch.LongTensor(self.bodyparts))
        init_A_select_inv = torch.transpose(init_A_select, -2, -1)

        init_dirs = init_kinect_joints[self.bones[:, 1]] - init_kinect_joints[self.bones[:, 0]]
        init_dirs = torch.tensor(init_dirs, dtype=torch.float32)
        init_dirs_A_inv = torch.bmm(init_A_select_inv[self.bones[:, 0]], init_dirs.unsqueeze(-1))[:, :, 0]

        target_dirs = target_kinect_joints[self.bones[:, 1]] - target_kinect_joints[self.bones[:, 0]]
        target_dirs = torch.tensor(target_dirs, dtype=torch.float32)
        target_dirs_A_inv = torch.bmm(init_A_select_inv[self.bones[:, 0]], target_dirs.unsqueeze(-1))[:, :, 0]

        twists = rotate_a_b_axis_angle_torch_batched(
            init_dirs_A_inv,
            target_dirs_A_inv
        )
        return twists

    def get_twists(self, A, target_kinect_joints, init_kinect_joints):
        # calculate kinect twists
        # get matrix A, corresponding to the proper bodypart
        A = A.clone().detach().cpu()
        a = torch.index_select(A.squeeze(0), 0, torch.LongTensor(self.bodyparts))
        a_inv = torch.inverse(a[:, :3, :3])

        kinect_twists = calc_kinect_twists(target_kinect_joints, init_kinect_joints, a_inv, self.bones)

        kinect_twists = universe_convert(np.array(kinect_twists), 'aa', 'rotmtx').reshape(31, -1)
        return kinect_twists

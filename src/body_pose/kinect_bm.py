import numpy as np
import torch
import time


def to_t(d, device, dtype=torch.float32):
    return torch.tensor(d, device=device, dtype=dtype)


def to_batch_t(x, device, size=None):
    if x is None:
        if size is None:
            raise Exception()
        x = np.zeros(size)
    return to_t(x, device=device).reshape(1, -1)


from patched_smplx.lbs import blend_shapes, batch_rodrigues, \
    batch_rigid_transform


def vertices2joints(J_regressor, vertices):
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def lbs(device, J, pose, parents, v_shaped,
        lbs_weights, dtype=torch.float32):
    """
    adapted from
    https://github.com/vchoutas/smplx/blob/master/smplx/lbs.py
    """

    batch_size = 1

    rot_mats = batch_rodrigues(
        pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

    # skip pose blend shapes
    # v_posed = pose_offsets + v_shaped
    v_posed = v_shaped

    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    W = lbs_weights[-32:].unsqueeze(dim=0).expand([batch_size, -1, -1])
    num_joints = len(parents)
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)

    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)[:, -32:]
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, A


class KinectJointsBM:
    def __init__(self, exp_bm_wrapper, gender, betas):
        self.bm = exp_bm_wrapper.pykinect_data[gender]['exp_bm']
        self.device = exp_bm_wrapper.device

        self.betas = to_batch_t(betas, self.device, 10)
        expression = to_batch_t(None, self.device, 10)
        shape_components = torch.cat(
            [self.betas, self.bm.fe_scale * expression], dim=-1)
        shapedirs = torch.cat([self.bm.shapedirs, self.bm.exprdirs], dim=-1)
        self.v_shaped = self.bm.v_template + blend_shapes(
            shape_components, shapedirs)
        self.J = vertices2joints(self.bm.J_regressor, self.v_shaped)

        self.pose_jaw = to_batch_t(None, self.device, 3)
        self.pose_eye = to_batch_t(None, self.device, 6)

        left_hand_pose = self.bm.left_hand_mean.reshape(1, -1).repeat(1, 1)
        right_hand_pose = self.bm.right_hand_mean.reshape(1, -1).repeat(1, 1)
        self.pose_hand = torch.cat([left_hand_pose, right_hand_pose], dim=1)

    def inf(self, pose_body, global_rot, global_trans):
        pose_body = to_batch_t(pose_body, self.device)
        global_rot = to_batch_t(global_rot, self.device)
        global_trans = to_batch_t(global_trans, self.device)

        full_pose = torch.cat(
            [global_rot, pose_body, self.pose_jaw, self.pose_eye, self.pose_hand],
            dim=1)

        verts, joints, A = lbs(
            device=self.device,
            J=self.J,
            pose=full_pose,
            parents=self.bm.kintree_table[0].long(),
            v_shaped=self.v_shaped,
            lbs_weights=self.bm.weights,
            dtype=self.bm.dtype
        )

        verts = verts + global_trans.unsqueeze(dim=1)

        joints = joints + global_trans.unsqueeze(dim=1)

        return verts, joints, A


def kinect_bm_out2kinect_joints(verts, joints, substract_pelvis=False):
    kinect_joints = verts[0, -32:].detach().cpu().numpy()

    if substract_pelvis:
        kinect_joints -= kinect_joints[[0], :]

    smplx_pelvis = joints[0, 0].detach().cpu().numpy()
    return kinect_joints, smplx_pelvis

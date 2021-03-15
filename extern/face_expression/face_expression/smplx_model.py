import torch
import torch.nn as nn
import numpy as np

from patched_smplx.vertex_ids import vertex_ids
from patched_smplx.vertex_joint_selector import VertexJointSelector
from patched_smplx.lbs import lbs, find_dynamic_lmk_idx_and_bcoords, vertices2landmarks, batch_rodrigues

from face_expression.index_mapping import (
    get_kinect_smpl_joints, get_kinect_smplx_vert, get_face_smplx_landmarks,
    get_shifted_op_joint_ids, get_smplx_ids_op_face
)


new_face_verts_ids = np.array([ 2210,  1962,  8835,  8733,  8774,  8740,  8770,  8746,  8765,
        9050,  9045,  9039,  9102,  9030,  2004,   567,   728,  9215,
        3157,   335,  3154,  2178,   673,  2135,    16,  2138,  9344,
        8997,  9005,  8945,  9008,  2751,  2793,  9000,  1676,  1623,
        2421,  9944,  9983,  2509, 10012, 10423,  1358,  9401,  9694,
        1135,  9773,  9494,  2827,  2813,  2774,  8987,  1657,  1696,
        1710,  1795,  1866,  8955,  2949,  2898,  2835,  2785,  8977,
        1668,  1718,  1827,  8953,  2929])


def shift_j_abs(body, joints_output, comp_device, kinect_smpl, shifts):
    v_num = kinect_smpl.shape[0]
    if shifts is None:
        body.kjj = joints_output[:, kinect_smpl[:, 1], :]
    else:
        n = joints_output.shape[0]
        shift_j = shifts[0:v_num, :, 0].reshape(1, -1, 3, 1).repeat(n, 1, 1, 1)
        j_h = torch.cat([shift_j, torch.ones((n, v_num, 1, 1), dtype=torch.float, requires_grad=False, device=comp_device)], dim=2)
        j_h = j_h.reshape(n, v_num, 1, 4)
        j_shift = torch.matmul(j_h, body.A[:,  kinect_smpl[:, 2], :, :]).reshape(n, v_num, 4)[:, :, 0:3]
#         for i in range(0, 100):
#             print('shape kin {} kin {} j_shift {} '.format(kinect_smpl.shape[0], joints_output.shape[1], j_shift.shape[1]))
        jos = joints_output[:, kinect_smpl[:, 1], :]
        body.kjj = jos + j_shift
#         exit()

def shift_op_j_abs(joints, A, shifts):
    if shifts is None:
        return
    inds = get_shifted_op_joint_ids()
    inds = np.asarray(inds, dtype=np.int) #14
    inds_t = torch.tensor(inds, dtype=torch.long, requires_grad=False, device=joints.device)
    b = joints.shape[0]
    shift_j = shifts.reshape(1, -1, 3, 1).repeat(b, 1, 1, 1)
    j_num = inds.shape[0]
    j_h = torch.cat([shift_j, torch.ones((b, j_num, 1, 1), dtype=torch.float, requires_grad=False, device=joints.device)], dim=2)
    j_h = j_h.reshape(b, j_num, 1, 4)
    j_shift = torch.matmul(j_h, A[:,  inds_t, :, :]).reshape(b, j_num, 4)[:, :, 0:3]
    joints[:, inds_t] = joints[:, inds_t] + j_shift
    
def shift_v_abs(body, verts_output, comp_device, kinect_smplvert, shifts):
    v_num = kinect_smplvert.shape[0]
    if shifts is None:
        body.kjv = verts_output[:, kinect_smplvert[:, 1], :]
    else:
        n = verts_output.shape[0]
        shift_v = shifts[-v_num:, :, 0].reshape(1, -1, 3, 1).repeat(n, 1, 1, 1)
        v_h = torch.cat(
            [shift_v, torch.ones((n, v_num, 1, 1), dtype=torch.float, requires_grad=False, device=comp_device)], dim=2)
        v_h = v_h.reshape(n, v_num, 1, 4)
        v_shift = torch.matmul(v_h, body.A[:, kinect_smplvert[:, 2], :, :]).reshape(n, v_num, 4)[:, :, 0:3]

    # joints_output[:, kinect_smpl[:, 1], :] = joints_output[:, kinect_smpl[:, 1], :] + torch.matmul(
    #     shifts[0:v_num, :, :].reshape(1, -1, 3, 10).repeat(n, 1, 1, 1), A_h.reshape(n, -1, 10, 1)).reshape(n, -1, 3)

        body.kjv = verts_output[:, kinect_smplvert[:, 1], :] + v_shift




class ExpBodyModel(nn.Module):

    def __init__(self,
                 bm_path='',
                 params=None,
                 num_betas=10,
                 v_template=None,
                 num_expressions=10,
                 use_posedirs=True,
                 dtype=torch.float32,
                 comp_device=None,
                 is_hand_pca=False,
                 num_pca_comps=6):

        super(ExpBodyModel, self).__init__()

        '''
        :param bm_path: path to a SMPL model as pkl file
        :param num_betas: number of shape parameters to include.
                if betas are provided in params, num_betas would be overloaded with number of thoes betas
        :param batch_size: number of smpl vertices to get
        :param device: default on gpu
        :param dtype: float precision of the compuations
        :return: verts, trans, pose, betas
        '''
        # Todo: if params the batchsize should be read from one of the params

        self.dtype = dtype

        if params is None: params = {}

        # -- Load SMPL params --
        if '.npz' in bm_path:
            smpl_dict = np.load(bm_path, encoding='latin1')
        else:
            raise ValueError('bm_path should be either a .pkl nor .npz file')

        njoints = smpl_dict['posedirs'].shape[2] // 3

        # Mean template vertices
        if v_template is None:
            v_template = np.repeat(smpl_dict['v_template'][np.newaxis], 1, axis=0)
        else:
            v_template = np.repeat(v_template[np.newaxis], 1, axis=0)

        self.register_buffer('v_template', torch.tensor(v_template, dtype=dtype))

        self.register_buffer('f', torch.tensor(smpl_dict['f'].astype(np.int32), dtype=torch.int32))

        if len(params):
            if 'betas' in params.keys():
                num_betas = params['betas'].shape[1]

        num_total_betas = smpl_dict['shapedirs'].shape[-1]
        if num_betas < 1:
            num_betas = num_total_betas

        shapedirs = smpl_dict['shapedirs'][:, :, :num_betas]
        self.register_buffer('shapedirs', torch.tensor(shapedirs, dtype=dtype))

        begin_shape_id = 300 if smpl_dict['shapedirs'].shape[-1] > 300 else 10        
        exprdirs = smpl_dict['shapedirs'][:, :, begin_shape_id:(begin_shape_id + num_expressions)]
        self.register_buffer('exprdirs', torch.tensor(exprdirs, dtype=dtype))

        # Regressor for joint locations given shape - 6890 x 24
        self.register_buffer('J_regressor', torch.tensor(smpl_dict['J_regressor'], dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 207, reshaped to 6890*30 x 207
        if use_posedirs:
            posedirs = smpl_dict['posedirs']
            posedirs = posedirs.reshape([posedirs.shape[0] * 3, -1]).T
            self.register_buffer('posedirs', torch.tensor(posedirs, dtype=dtype))
        else:
            self.posedirs = None

        # indices of parents for each joints
        kintree_table = smpl_dict['kintree_table'].astype(np.int32)
        self.register_buffer('kintree_table', torch.tensor(kintree_table, dtype=torch.long, device=comp_device))

        # LBS weights
        # weights = np.repeat(smpl_dict['weights'][np.newaxis], batch_size, axis=0)
        weights = smpl_dict['weights']
        self.register_buffer('weights', torch.tensor(weights, dtype=dtype))


        lmk_faces_idx = smpl_dict['lmk_faces_idx']
        self.lmk_faces_idx = torch.tensor(lmk_faces_idx, dtype=torch.long, device=comp_device)
        lmk_bary_coords = smpl_dict['lmk_bary_coords']
        self.lmk_bary_coords = torch.tensor(lmk_bary_coords, dtype=torch.float, device=comp_device)
        dynamic_lmk_faces_idx = smpl_dict['dynamic_lmk_faces_idx']
        self.dynamic_lmk_faces_idx = torch.tensor(dynamic_lmk_faces_idx, dtype=torch.long, device=comp_device)

        dynamic_lmk_bary_coords = smpl_dict['dynamic_lmk_bary_coords']
        self.dynamic_lmk_bary_coords = torch.tensor(
            dynamic_lmk_bary_coords, dtype=dtype, device=comp_device)

        self.neck_kin_chain = []
        NECK_IDX = 12
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long, device=comp_device)
        parents = self.kintree_table[0]
        while curr_idx != -1:
            self.neck_kin_chain.append(curr_idx)
            curr_idx = parents[curr_idx]

        self.neck_kin_chain = torch.stack(self.neck_kin_chain)

        # SMPL and SMPL-H share the same topology, so any extra joints can
        # be drawn from the same place

        self.dtype = dtype

        self.vertex_joint_selector = VertexJointSelector(
            vertex_ids=vertex_ids['smplx'])

        self.is_hand_pca = is_hand_pca
        self.num_pca_comps = num_pca_comps
        if is_hand_pca:
            self.left_hand_mean = torch.tensor(smpl_dict['hands_meanl'], dtype=torch.float, device=comp_device,
                                               requires_grad=False)
            self.right_hand_mean = torch.tensor(smpl_dict['hands_meanr'], dtype=torch.float, device=comp_device,
                                                requires_grad=False)
            self.left_hand_components = torch.tensor(smpl_dict['hands_componentsl'][:num_pca_comps], dtype=torch.float,
                                                     device=comp_device, requires_grad=False)
            self.right_hand_components = torch.tensor(smpl_dict['hands_componentsr'][:num_pca_comps], dtype=torch.float,
                                                      device=comp_device, requires_grad=False)

        self.kinect_smpl = get_kinect_smpl_joints()
        self.kinect_smplvert = get_kinect_smplx_vert()
        self.face_smplx = get_face_smplx_landmarks()


    def forward(self, root_orient, pose_body, pose_hand, pose_jaw, pose_eye, betas,
                trans, expression, shifts=None, shifts_op=None, return_dict=False, **kwargs):
        '''

        :param root_orient: Nx3
        :param pose_body:
        :param pose_hand:
        :param pose_jaw:
        :param pose_eye:
        :param kwargs:
        :return:
        '''
        # assert not (v_template  is not None and betas  is not None), ValueError('vtemplate and betas could not be used jointly.')

        if self.is_hand_pca:
            n_h = int(pose_hand.shape[1] / 2)
            left_hand_pose = pose_hand[:, 0:n_h]
            right_hand_pose = pose_hand[:, n_h:]
            b = pose_hand.shape[0]
            left_hand_pose = self.left_hand_mean.reshape(1, -1).repeat(b, 1) + torch.einsum(
                'bi,ij->bj', [left_hand_pose, self.left_hand_components])
            right_hand_pose = self.right_hand_mean.reshape(1, -1).repeat(b, 1) + torch.einsum(
                'bi,ij->bj', [right_hand_pose, self.right_hand_components])
            pose_hand = torch.cat([left_hand_pose, right_hand_pose], dim=1)

        v_template = self.v_template
        if betas is None: betas = self.betas

        full_pose = torch.cat([root_orient, pose_body, pose_jaw, pose_eye, pose_hand],
                                  dim=1)  # orient:3, body:63, jaw:3, eyel:3, eyer:3, handl, handr

        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)

        verts, joints, A, joints_init = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                                            shapedirs=shapedirs, posedirs=self.posedirs.unsqueeze(0),
                                            J_regressor=self.J_regressor, parents=self.kintree_table[0].long(),
                                            lbs_weights=self.weights,
                                            dtype=self.dtype)
        
        batch_size = full_pose.size(0)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3), dtype=self.dtype).view([batch_size, -1, 3, 3])[:, 0]

        bs = shape_components.shape[0]
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(
            dim=0).expand(bs, -1).contiguous()
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).repeat(
            bs, 1, 1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = find_dynamic_lmk_idx_and_bcoords(
            verts, full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)

        lmk_faces_idx = torch.cat([lmk_faces_idx,
                                   dyn_lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat(
            [lmk_bary_coords.expand(bs, -1, -1),
             dyn_lmk_bary_coords], 1)

        landmarks = vertices2landmarks(verts, self.f.long(),
                                       lmk_faces_idx,
                                       lmk_bary_coords)

        # Add any extra joints that might be needed
        joints = self.vertex_joint_selector(verts, joints)
        # Add the landmarks to the joints
        joints = torch.cat([joints, landmarks], dim=1)
        # Map the joints to the current dataset

        # if self.joint_mapper is not None:
        #     joints = self.joint_mapper(joints=joints, vertices=verts)

        Jtr = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)
        
        index = get_smplx_ids_op_face()[:68]
        Jtr[:, index[17:36]] = verts[:, new_face_verts_ids[17:36]]

        return Jtr[:, get_smplx_ids_op_face()[:68]], rot_mats, verts
    
    
    
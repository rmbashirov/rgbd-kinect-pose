import os
import os.path as osp
import numpy as np
import pickle
from copy import deepcopy
import cv2
import yaml
from collections import defaultdict
import time

import torch
from torch import nn

from smplx_optimization.pykinect.mymath import orthoprocrustes

from smplx_kinect.common.exp_bm_wrapper import ExpBMWrapper, exp_bm_out2kinect_joints
from smplx_kinect.common.concater import NamedDataConcater, universe_len
from smplx_kinect.common.angle_representation import universe_convert
from smplx_kinect.exp.net.config2net import get_net
from smplx_kinect.exp.kp_processor import KPProcessor


from .kinect_bm import KinectJointsBM, kinect_bm_out2kinect_joints


class GlobalPositioner:
    def __init__(self):
        self.filtered_kinect_joints = np.array([
            0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 29, 31])

    def fit(self, target_points, fit_points, center):
        """
        fit_points of shape (n, 3)
        target_points of shape (n, 3)
        center of shape (3,)
        """

        R, t = orthoprocrustes(
            fit_points[self.filtered_kinect_joints] - center.reshape(1, -1),
            target_points[self.filtered_kinect_joints]
        )
        rvec = cv2.Rodrigues(R)[0][:, 0]
        # tvec = t
        tvec = t - center
        return rvec, tvec


class Inferer:
    def __init__(self, model_dp, checkpoint, device, beta, gender, pykinect_data_dp, fast=True):
        with open(osp.join(model_dp, 'config.yaml'), 'r') as f:
            self.train_config = yaml.safe_load(f)

        self.device = device
        self.exp_bm_wrapper = ExpBMWrapper(
            pykinect_data_dp=pykinect_data_dp,
            # device=self.device
            device='cpu'
        )

        self.is_residual_output = self.train_config['net_output']['pose_init_residual']

        self.kp_processor = KPProcessor(
            self.train_config['kinect_kp_processor']['val'])

        train_sample_fp = osp.join(
            model_dp, 'train_sample_azure.pickle')
        with open(train_sample_fp, 'rb') as f:
            # ndc_train_sample = renamed_load(f)
            ndc_train_sample = pickle.load(f)

        self.train_config['load_params_fp'] = osp.join(
            model_dp, 'models', f'net_{checkpoint:06d}')
        self.train_config['load_param_exclude'] = None
        self.net = get_net(self.train_config, ndc_train_sample, device)

        self.net_input = self.train_config['net_input']
        self.seq_len = 1
        self.beta = beta
        self.gender = gender
        self.device = device
        self.global_positioner = GlobalPositioner()
        self.output_count = 0
        self.hidden = None

        self.fast = fast
        self.kj_bm = KinectJointsBM(
            self.exp_bm_wrapper,
            gender=self.gender,
            betas=self.beta
        )

        self.timings = defaultdict(list)

    def inf(self, input_kinect_joints, kinect_confs):
        input_kinect_joints_raw = deepcopy(input_kinect_joints)

        input_kinect_joints = self.kp_processor.process(
            input_kinect_joints, meta=None
        )

        net_input = defaultdict(list)

        if self.net_input['kinect_kp']:
            net_input['kinect_kp'].append(input_kinect_joints)

        if self.net_input['beta']:
            net_input['beta'].append(self.beta)

        if self.net_input['pose_init'] or self.net_input['twists']:
            init_body_pose, init_global_rot, init_global_trans = \
                self.exp_bm_wrapper.get_smplx_init(
                    input_kinect_joints,
                    np.ones(len(input_kinect_joints)) * 2,
                    self.beta, self.gender)

            if self.net_input['pose_init']:
                net_input['pose_init'].append(init_body_pose)

            if self.net_input['twists']:
                if self.fast:
                    verts, joints, A = self.kj_bm.inf(
                        pose_body=init_body_pose,
                        global_rot=init_global_rot,
                        global_trans=init_global_trans
                    )
                    kinect_joints, _ = kinect_bm_out2kinect_joints(
                        verts, joints, substract_pelvis=True)
                    twists = self.exp_bm_wrapper.get_twists_v2(
                        init_A=A,
                        init_kinect_joints=kinect_joints,
                        target_kinect_joints=input_kinect_joints
                    )
                    twists = twists.detach().cpu().numpy()
                else:
                    init_exp_bm_out = self.exp_bm_wrapper.inf_exp_bm(
                        gender=self.gender,
                        beta=self.beta,
                        pose_body=init_body_pose,
                        global_rot=init_global_rot,
                        global_trans=init_global_trans
                    )
                    init_kinect_joints, _ = exp_bm_out2kinect_joints(
                        init_exp_bm_out,
                        substract_pelvis=True,
                        return_smplx_pelvis=False
                    )
                    twists = self.exp_bm_wrapper.get_twists_v2(
                        init_A=init_exp_bm_out.A,
                        init_kinect_joints=init_kinect_joints,
                        target_kinect_joints=input_kinect_joints
                    )
                    twists = twists.detach().cpu().numpy()

                net_input['twists'].append(twists)

        for k in net_input:
            net_input[k] = np.stack(net_input[k])

        # self.add_timing('0', (time.time() - start) * 1000)

        # train_sample2ndc
        net_input_ndcs = []
        for k in self.net_input:
            if self.net_input[k]:
                d = net_input[k]
                d = d.reshape(self.seq_len, -1)
                if k in ['pose_init', 'twists']:
                    d = universe_convert(d,
                        'aa', self.train_config['input_angle_representation'])
                ndc = NamedDataConcater(d, k, is_seq=True)
                net_input_ndcs.append(ndc)
        net_input_ndc = NamedDataConcater.concat(
            net_input_ndcs, name='net_input')

        # collate_fn
        net_input_batch = []
        transform_f = lambda x: torch.tensor(x, dtype=torch.float32)
        net_input_ndc.transform_data(transform_f)
        net_input_batch.append(net_input_ndc)
        stack_f = lambda x: torch.stack(x, dim=0)
        net_input_batch = NamedDataConcater.stack(net_input_batch, stack_f)
        transform_f = lambda x: x.to(device=self.device)
        net_input_batch.transform_data(transform_f)

        net_out_dict, self.hidden = self.net(net_input_batch.data, self.hidden)
        net_out_body_pose = net_out_dict['body_pose']
        B = net_out_body_pose.shape[0]
        F = net_out_body_pose.shape[2]
        S = net_out_body_pose.shape[1]

        if self.is_residual_output:
            pose_init = net_input_batch.name2data('pose_init')
            assert pose_init.shape == net_out_body_pose.shape
            residual = net_out_body_pose
            shape = B * S * 21, 3, 3
            pose_init = pose_init.reshape(shape)
            residual = residual.reshape(shape)
            pred = torch.bmm(residual, pose_init)
        else:
            pred = net_out_body_pose

        body_pose_aa = universe_convert(
            pred, self.train_config['input_angle_representation'], 'aa')
        body_pose = body_pose_aa.detach().cpu().numpy().reshape(-1)

        if self.fast:
            verts, joints, _ = self.kj_bm.inf(
                pose_body=body_pose,
                global_rot=np.zeros(3),
                global_trans=np.zeros(3))
            result_kinect_joints, smplx_pelvis = kinect_bm_out2kinect_joints(
                verts, joints, substract_pelvis=False)
        else:
            result_exp_bm_out = self.exp_bm_wrapper.inf_exp_bm(
                gender=self.gender,
                beta=self.beta,
                pose_body=body_pose
            )
            result_kinect_joints, _, smplx_pelvis = exp_bm_out2kinect_joints(
                result_exp_bm_out,
                substract_pelvis=True,
                return_smplx_pelvis=True
            )

        if not hasattr(self, 'index'):
            self.index = 0
        else:
            self.index += 1

        global_rot, global_trans = self.global_positioner.fit(
            input_kinect_joints_raw,
            result_kinect_joints,
            smplx_pelvis
        )

        return {
            'input_kinect_joints': input_kinect_joints,
            'body_pose': body_pose,
            'global_rot': global_rot,
            'global_trans': global_trans
        }

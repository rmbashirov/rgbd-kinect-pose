import os
import os.path as osp
import datetime
import argparse
import yaml
import numpy as np
import torch
import pickle
import operator
from copy import deepcopy


import torch


from smplx_kinect.common.angle_representation import universe_convert
from smplx_kinect.common.metrics import MetricsEngine
from smplx_kinect.common.body_models import SMPLXWrapper

from body_pose.inferer import Inferer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_config')
    parser.add_argument('--metrics_config')
    args = parser.parse_args()
    return args


class BodyPose:
    """
    non multiprocessing_pipeline version of blocks/body_pose.py
    """
    def __init__(self, data_dirpath, processor_config, pykinect_data_dp, fast, beta, gender):
        self.data_dirpath = data_dirpath
        self.processor_config = processor_config
        self.pykinect_data_dp = pykinect_data_dp
        self.log_level = processor_config.get('log_level', 2)
        self.device = torch.device(self.processor_config['device'])

        self.gender = gender
        self.beta = beta
        self.model_dp = osp.join(self.data_dirpath,
                                 self.processor_config['model_path'])

        self.inferer = Inferer(
            model_dp=self.model_dp,
            checkpoint=self.processor_config['checkpoint'],
            device=self.device,
            beta=self.beta,
            gender=self.gender,
            pykinect_data_dp=self.pykinect_data_dp,
            fast=fast
        )

    def kinect_joints2body_pose(self, kinect_joints):
        inf_result = self.inferer.inf(kinect_joints, kinect_confs=None)
        return inf_result


class Dataset:
    def __init__(self, config):
        self.config = config

        from azure_unpacked import AzureUnpackedData

        self.sns = self.config['sns']

        azure_unpacked_dp = self.config['azure_unpacked_dp']
        self.auds = dict()
        self.smplx_fit_data = dict()
        for seq in self.config['seqs']:
            aud = AzureUnpackedData(
                osp.join(azure_unpacked_dp, seq),
                serial_numbers=self.sns
            )
            aud.read_bt()
            aud.parse_cam_params()
            self.auds[seq] = aud

            self.smplx_fit_data[seq] = dict()
            for sn in self.sns:
                with open(osp.join(self.config['converted_fit_dp'], seq, f'{sn}.pickle'), 'rb') as f:
                    loaded_data = pickle.load(f)
                    self.smplx_fit_data[seq][sn] = loaded_data


def calc_metrics(server_config, metrics_config, args):
    pykinect_data_dp = osp.join(server_config['data_dirpath'], 'pykinect')

    smplx_wrapper = SMPLXWrapper(metrics_config['body_models_dp'], device='cpu')
    metrics_engine = MetricsEngine(smplx_wrapper)

    dataset = Dataset(metrics_config)

    seqs = metrics_config['seqs']
    sns = metrics_config['sns']

    result_metrics = dict()

    for seq in seqs:
        smplx_fit_data = dataset.smplx_fit_data[seq][sns[0]]
        beta = smplx_fit_data['betas']
        gender = 'male' if smplx_fit_data['is_male'] else 'female'

        body_pose = BodyPose(
            data_dirpath=server_config['data_dirpath'],
            processor_config=server_config['body_pose'],
            pykinect_data_dp=pykinect_data_dp,
            fast=metrics_config['fast'],
            beta=beta,
            gender=gender
        )

        for sn in sns:
            smplx_fit_data = dataset.smplx_fit_data[seq][sn]
            metrics_engine.reset()

            loop_data = list(sorted(
                smplx_fit_data['frame_index2fit_index'].items(),
                key=operator.itemgetter(0)
            ))  # [metrics_config['skip']:]

            loop_index = 0
            for frame_id, fit_index in loop_data:
                bt_res = dataset.auds[seq].get_bt_joints(
                    sn, frame_id, return_confs=True, d2rgb=True)
                if bt_res is not None:
                    kinect_joints, kinect_confs = bt_res

                    pred_body_pose = body_pose.kinect_joints2body_pose(
                        kinect_joints)['body_pose']

                    if loop_index == 0:
                        print(f'loop {frame_id}')
                        for i in range(30):
                            pred_body_pose = body_pose.kinect_joints2body_pose(
                                kinect_joints)['body_pose']

                    target_body_pose = smplx_fit_data['body_pose'][fit_index]

                    def convert(aa):
                        eye = np.eye(3)
                        result = universe_convert(aa, 'aa', 'rotmtx').reshape(-1, 3, 3)
                        result = np.concatenate((eye[None, :, :], result)).reshape(1, 1, -1)
                        # result = np.expand_dims(result, 1)
                        return result

                    target_body_pose = convert(target_body_pose)  # 1x1x198
                    pred_body_pose = convert(pred_body_pose)

                    metrics = metrics_engine.compute(
                        pred_body_pose,
                        target_body_pose
                    )
                    metrics_engine.aggregate(metrics)
                    loop_index += 1
            summary_metrics = metrics_engine.get_summary_metrics([1])
            # summary_metrics_str = metrics_engine.get_summary_metrics_str(summary_metrics)
            # print(summary_metrics)
            result_metrics[f'{seq}_{sn}'] = deepcopy(summary_metrics)

    s = '=' * 10
    for m in ['auc', 'euler', 'joint_angle', 'positional']:
        values = dict()
        for k in result_metrics:
            values[k] = result_metrics[k][1][m]
        value = np.mean(list(values.values()))
        s += f'\n{m}: {value:.3f}'
        for k in result_metrics:
            s += f'\n\t{k}: {values[k]:.3f}'
    print(s)

    output_dp = metrics_config['output_dp']
    os.makedirs(output_dp, exist_ok=True)

    p1 = osp.basename(server_config['body_pose']['model_path'])
    p2 = '{:06d}'.format(server_config['body_pose']['checkpoint'])
    p3 = osp.splitext(osp.basename(args.metrics_config))[0]

    fp = osp.join(output_dp, f'{p1}_{p2}_{p3}.txt')
    with open(fp, 'w') as f:
        f.write(s)


def main():
    args = parse_args()
    with open(args.server_config, 'r') as f:
        server_config = yaml.safe_load(f)

    with open(args.metrics_config, 'r') as f:
        metrics_config = yaml.safe_load(f)

    # output_dirpath = osp.join(
    #     server_config['output_dirpath'],
    #     datetime.datetime.strftime(
    #         datetime.datetime.now(),
    #         "%Y.%m.%d_%H:%M:%S"))
    # os.makedirs(output_dirpath, exist_ok=True)

    calc_metrics(server_config, metrics_config, args)


if __name__ == "__main__":
    main()

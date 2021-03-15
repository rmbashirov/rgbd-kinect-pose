import os.path as osp
import numpy as np
from collections import defaultdict
import tqdm

from ..dummy_data import DummyDict

from smplx_kinect.common.exp_bm_wrapper import exp_bm_out2kinect_joints


class AzureTrainDataset:
    def __init__(self,
                 seq_dataset,
                 exp_bm_wrapper,
                 net_input,
                 kp_processor=None,
                 fast_features_loader=None):
        self.name = 'azure'
        self.seq_dataset = seq_dataset
        self.exp_bm_wrapper = exp_bm_wrapper
        self.net_input = net_input
        self.kp_processor = kp_processor
        self.dummy_dict = DummyDict()
        self.fast_features_loader = fast_features_loader

    def __len__(self):
        return len(self.seq_dataset)

    def __getitem__(self, index):
        # # code from AzureSeqLoader
        # raw_train_sample = {
        #     'global_rot': d['global_rot'][fix_sample_info.slice],
        #     'global_trans': d['global_trans'][fix_sample_info.slice],
        #     'kinect_joints': np.array([
        #       d['kinect_joints'][i] is not None for i in slice_ids]),
        #     'kinect_confs': np.array([
        #       d['kinect_confs'][i] is not None for i in slice_ids]),
        #     'body_pose': d['body_pose'][fix_sample_info.slice],
        #     'betas': d['betas'],
        #     'gender': d['gender'],
        #     'fix_sample_info': fix_sample_info,
        # }
        raw_train_sample = self.seq_dataset[index]

        seq_len = len(raw_train_sample['global_rot'])

        beta = raw_train_sample['betas']
        gender = raw_train_sample['gender']

        net_input = defaultdict(list)
        net_target = []

        meta = {
            'fix_sample_info': raw_train_sample['fix_sample_info'],
            'frame_ids': raw_train_sample['frame_ids'],
            'gender': raw_train_sample['gender'],
            'valid': raw_train_sample['valid'],
            'seq_len': seq_len,
            'ds': 'azure'
        }

        fast_features = None
        if self.net_input['pose_init'] or self.net_input['twists']:
            if self.fast_features_loader is not None:
                path = raw_train_sample['fix_sample_info'].fp()
                fast_features = self.fast_features_loader.get(path)

        # for i in tqdm.tqdm(range(seq_len)):
        for i in range(seq_len):
            input_kinect_joints = raw_train_sample['kinect_joints'][i]
            if self.kp_processor is not None:
                input_kinect_joints = self.kp_processor.process(
                    input_kinect_joints, meta=meta
                )
            valid = raw_train_sample['valid'][i]
            if not valid:
                raise Exception()
            if self.net_input['kinect_kp']:
                net_input['kinect_kp'].append(input_kinect_joints)
            if self.net_input['beta']:
                net_input['beta'].append(beta)

            if self.net_input['pose_init'] or self.net_input['twists']:
                # valid is already handled in fast features
                if fast_features is not None:
                    for k in [
                        k for k in ['pose_init', 'twists']
                        if self.net_input[k]
                    ]:
                        frame_id = meta['frame_ids'][i]
                        ffi = fast_features['frame_ids'].index(frame_id)
                        net_input[k].append(fast_features[k][ffi])
                else:
                    # some kp do not exist for specific frame
                    if valid:
                        init_body_pose, init_global_rot, init_global_trans = \
                            self.exp_bm_wrapper.get_smplx_init(
                                input_kinect_joints,
                                np.ones(len(input_kinect_joints)) * 2,
                                beta, gender)
                        if self.net_input['pose_init']:
                            self.dummy_dict.set('pose_init', init_body_pose)
                            net_input['pose_init'].append(init_body_pose)
                        if self.net_input['twists']:
                            init_exp_bm_out = self.exp_bm_wrapper.inf_exp_bm(
                                gender=gender,
                                beta=beta,
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
                            self.dummy_dict.set('twists', twists)
                            net_input['twists'].append(twists)
                    else:
                        if self.net_input['pose_init']:
                            net_input['pose_init'].append(
                                self.dummy_dict.get('pose_init'))
                        if self.net_input['twists']:
                            net_input['twists'].append(
                                self.dummy_dict.get('twists'))
            net_target.append(raw_train_sample['body_pose'][i])

        try:
            train_sample = {
                'net_input': {
                    k: np.stack(net_input[k])
                    for k in net_input.keys()
                },
                'net_target': np.stack(net_target),
                'meta': meta
            }
        except Exception as e:
            for i in range(seq_len):
                frame_id = meta['frame_ids'][i]
                ffi = fast_features['frame_ids'].index(frame_id)
                print(fast_features is not None)
                if fast_features is not None:
                    for k in [
                        k for k in ['pose_init', 'twists']
                        if self.net_input[k]
                    ]:
                        print(ffi)
                        print(fast_features[k][ffi].shape)
                else:
                    print('azazaza')

            for k in net_input.keys():
                print(k)
                for el in net_input[k]:
                    print(el.shape)

            raise e

        return train_sample

import os.path as osp
import pickle
import tqdm
import numpy as np
from collections import defaultdict
from copy import deepcopy

from ..dummy_data import DummyDict


class SampleInfo:
    def __init__(self, person_id, sn, slice_name):
        self.name = 'azure'
        self.person_id = person_id
        self.sn = sn
        self.slice_name = slice_name

    def __str__(self):
        return f'{self.name}_{self.person_id}_{self.sn}_{self.slice_name}'

    def fp(self):
        """
        Fast features filepath
        """
        return f'{self.name}/{self.person_id}/{self.sn}.pickle'


class AzureLoader:
    def __init__(self,
                 kinect_unpacked_dp,
                 person_ids, sns, converted_fit_dp,
                 kinect_unpacked_dp_bt=None):
        from azure_unpacked import AzureUnpackedData

        self.kinect_unpacked_dp = kinect_unpacked_dp
        self.kinect_unpacked_dp_bt = kinect_unpacked_dp_bt
        self.person_ids = person_ids
        self.sns = sns

        self.auds = dict()
        self.smplx_fit_data = dict()

        tq = tqdm.tqdm(total=len(self.person_ids))
        tq.set_description(f'Azure loader')
        tq.refresh()
        for person_id in self.person_ids:
            tq.set_postfix({'person_id': person_id})
            tq.refresh()

            aud = AzureUnpackedData(
                osp.join(self.kinect_unpacked_dp, person_id),
                serial_numbers=sns
            )
            bt_dp = osp.join(self.kinect_unpacked_dp_bt, person_id) \
                if self.kinect_unpacked_dp_bt is not None else None
            aud.read_bt(dataset_dirpath=bt_dp)
            aud.parse_cam_params()
            self.auds[person_id] = aud
            # print(f'Azure load {person_id}, length:', len(aud.bt))

            self.smplx_fit_data[person_id] = dict()
            for sn in self.sns:
                # check loaded pickle data format in
                # "/Vol0/user/r.bashirov/datasets/AzurePeople/avakhitov_fits_converted/info.txt"
                with open(osp.join(converted_fit_dp, person_id, f'{sn}.pickle'), 'rb') as f:
                    loaded_data = pickle.load(f)
                    self.smplx_fit_data[person_id][sn] = loaded_data

            tq.update()
            tq.refresh()
        tq.close()


class AzureSeqLoader:
    def __init__(
            self,
            azure_loader, seq_len, max_part_len,
            fast_features_loader=None
    ):
        self.seq_len = seq_len
        self.max_part_len = max_part_len
        self.fast_features_loader = fast_features_loader

        self.samples_data = dict()
        self.samples_info = []
        self.max_try_count = 100
        self.dummy_dict = DummyDict()
        self.set_samples(azure_loader, max_part_len)

    def set_samples(self, azure_loader, max_part_len):
        for person_id in azure_loader.person_ids:
            for sn in azure_loader.sns:
                fit_data = azure_loader.smplx_fit_data[person_id][sn]
                frame_ids = fit_data['frame_index2fit_index'].keys()
                frame_ids = list(sorted(frame_ids))
                n = len(frame_ids)
                assert n == fit_data['n']

                frame_ids_splits = np.array_split(
                    frame_ids,
                    np.ceil(len(frame_ids) / max_part_len))
                fit_ids_splits = np.array_split(
                    np.arange(len(frame_ids)),
                    np.ceil(len(frame_ids) / max_part_len))

                assert len(frame_ids_splits) == len(fit_ids_splits)

                for part_frame_ids, part_fit_ids in zip(frame_ids_splits, fit_ids_splits):
                    assert len(part_frame_ids) == len(part_fit_ids)
                    n = len(part_frame_ids)
                    smplx_fit_data = azure_loader.smplx_fit_data[person_id][sn]
                    part_data = {
                        'global_rot': smplx_fit_data['global_rvec'][part_fit_ids],
                        'global_trans': smplx_fit_data['global_tvec'][part_fit_ids],
                        'body_pose': smplx_fit_data['body_pose'][part_fit_ids],
                        'betas': smplx_fit_data['betas'],
                        'n': n,
                        'gender': 'male' if smplx_fit_data['is_male'] else 'female'
                    }
                    upd = defaultdict(list)
                    for i in range(len(part_frame_ids)):
                        frame_id = part_frame_ids[i]
                        upd['frame_ids'].append(frame_id)

                        bt_res = azure_loader.auds[person_id].get_bt_joints(
                            sn, frame_id, return_confs=True, d2rgb=True)
                        if bt_res is not None:
                            kinect_joints, kinect_confs = bt_res
                            kinect_confs = (kinect_confs == 2).astype(np.float32)
                            self.dummy_dict.set('kinect_joints', kinect_joints)
                            self.dummy_dict.set('kinect_confs', kinect_confs)
                            upd['valid'].append(True)
                        else:
                            kinect_joints = self.dummy_dict.get('kinect_joints')
                            kinect_confs = self.dummy_dict.get('kinect_confs')
                            upd['valid'].append(False)
                        upd['kinect_joints'].append(kinect_joints)
                        upd['kinect_confs'].append(kinect_confs)

                        img_path = azure_loader.auds[person_id].get_color_undistorted(
                            sn, frame_id, get_path=True)
                        upd['img_path'].append(img_path)
                    part_data.update(upd)

                    slice_name = f'{part_frame_ids[0]:06d}_{part_frame_ids[-1]:06d}'
                    sample_info = SampleInfo(
                        person_id=person_id,
                        sn=sn,
                        slice_name=slice_name
                    )

                    if self.seq_len != 'full':
                        valid = np.array(deepcopy(part_data['valid']), dtype=np.int32)
                        # at least 1 frame with no kp exists
                        if np.sum(valid) <= len(valid):
                            i = 0
                            while i < n:
                                if valid[i] == 0:
                                    left = max(i - self.seq_len + 1, 0)
                                    # right = min(i + self.seq_len - 1, n)
                                    valid[left:i + 1] = 0
                                i += 1
                            if np.mean(valid) < 0.5:
                                # the result of get_slice can return None
                                print(f'Sample {sample_info} not valid: {np.mean(valid) * 100:.1f}%')
                                continue

                    self.samples_data[str(sample_info)] = part_data
                    self.samples_info.append(sample_info)

    def get_slice(self, d, sample_info, max_try_count=None):
        if self.seq_len == 'full':
            return slice(None, None)
        if max_try_count is None:
            max_try_count = self.max_try_count
        n = d['n']
        max_start = n - (self.seq_len + 1)
        if max_start < 1:
            return None
        try_count = 0
        while try_count < max_try_count:
            start = np.random.randint(0, max_start)
            result = slice(start, start + self.seq_len)
            # check all kinect joints exist in slice
            if not all([
                d['valid'][i]
                for i in np.arange(n)[result]
            ]):
                try_count += 1
            else:
                return result
        print(f'Exceeded number of tries for sample {sample_info}')
        return None

    def get_fix_sample_info(self, sample_info):
        if hasattr(sample_info, 'slice'):
            return sample_info
        d = self.samples_data[str(sample_info)]
        sl = self.get_slice(d, sample_info)
        assert sl is not None
        result_sample_info = deepcopy(sample_info)
        result_sample_info.slice = sl
        return result_sample_info

    def sample_info2raw_train_sample(self, sample_info):
        d = self.samples_data[str(sample_info)]
        fix_sample_info = self.get_fix_sample_info(sample_info)

        slice_ids = np.arange(d['n'])[fix_sample_info.slice]

        raw_train_sample = {
            'global_rot': d['global_rot'][fix_sample_info.slice],
            'global_trans': d['global_trans'][fix_sample_info.slice],
            'kinect_joints': np.array([
                d['kinect_joints'][i] for i in slice_ids], dtype=np.float32),
            'kinect_confs': np.array([
                d['kinect_confs'][i] for i in slice_ids], dtype=np.float32),
            'valid': np.array([
                d['valid'][i] for i in slice_ids], dtype=np.bool),
            'body_pose': d['body_pose'][fix_sample_info.slice],
            'betas': d['betas'],
            'gender': d['gender'],
            'fix_sample_info': fix_sample_info,
            'frame_ids': d['frame_ids'][fix_sample_info.slice]
        }
        return raw_train_sample

    def __len__(self):
        return len(self.samples_info)

    def __getitem__(self, index):
        sample_info = self.samples_info[index]
        raw_train_sample = self.sample_info2raw_train_sample(sample_info)
        return raw_train_sample

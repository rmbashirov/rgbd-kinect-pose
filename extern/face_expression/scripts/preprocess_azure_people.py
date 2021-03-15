import os
import json
import h5py
import pickle
from tqdm import tqdm

import numpy as np


image_root = "/Vol0/user/r.bashirov/workdir/git/data/smplx_kinect/test_capture/offline_processor2"
smplx_root = "/Vol1/dbstore/datasets/a.vakhitov/kinect_dataset/full_testcap_30"
openpose_root = "/Vol1/dbstore/datasets/k.iskakov/data/AzurePeopleTest/openpose"

h5_path = os.path.join(
    "/Vol1/dbstore/datasets/k.iskakov/data/AzurePeopleTest/meta", "AzurePeopleTest.test.h5"
)
scheme_path = os.path.join(
    "/Vol1/dbstore/datasets/k.iskakov/data/AzurePeopleTest/meta", "AzurePeopleTest_scheme.test.pkl"
)
    
# load scheme
scheme = []
            
# subject_ids = sorted(os.listdir(openpose_root))
subject_ids = ['04_simple']
for subject_id in tqdm(subject_ids):
    # camera_ids = sorted(os.listdir(os.path.join(openpose_root, subject_id)))
    # camera_ids = list(filter(lambda x: os.path.isdir(os.path.join(openpose_root, subject_id, x)), camera_ids))
    camera_ids = ['000062692912']
    
    for camera_id in camera_ids:
        identity_id = subject_id
        video_id = camera_id
        utterance_id = 0
        
        scheme.append((identity_id, video_id, utterance_id))

scheme = sorted(scheme)

# build h5 file
with h5py.File(h5_path, 'w') as hf:
    for (identity_id, video_id, utterance_id) in tqdm(scheme):
        # load smplx
        smplx_dir = os.path.join(smplx_root, identity_id, "mv", video_id, "joints_op_face_hand")

        expressions = np.load(os.path.join(smplx_dir, 'expressions.npy')) # * 100; TODO: maybe multiply by 100
        poses = np.load(os.path.join(smplx_dir, 'poses.npy'))
        betas = np.load(os.path.join(smplx_dir, 'betas.npy'))

        # load camera matrix
        mkv_meta_json_path = os.path.join(image_root, identity_id, video_id, "mkv_meta.json")
        with open(mkv_meta_json_path) as json_file:
            mkv_meta = json.load(json_file)
        
        camera_matrix = np.eye(3)
        camera_matrix[0, 0] = mkv_meta["rgb_undistorted_intrinsics"]['fx']
        camera_matrix[1, 1] = mkv_meta["rgb_undistorted_intrinsics"]['fy']
        camera_matrix[0, 2] = mkv_meta["rgb_undistorted_intrinsics"]['cx']
        camera_matrix[1, 2] = mkv_meta["rgb_undistorted_intrinsics"]['cy']

        # load openpose keypoints 2d
        openpose_dir = os.path.join(openpose_root, identity_id, video_id)

        face_keypoints_2d_list = []
        names = sorted(os.listdir(openpose_dir))
        for name in names:
            path = os.path.join(openpose_dir, name)

            with open(path) as f:
                openpose_data = json.load(f)
                
            if len(openpose_data['people']) > 0:
                face_keypoints_2d = openpose_data['people'][0]['face_keypoints_2d']
                face_keypoints_2d = np.array(face_keypoints_2d).reshape(70, 3)
                face_keypoints_2d = face_keypoints_2d[:, :2]  # remove confidences
            else:
                face_keypoints_2d = np.zeros((70, 2))

            face_keypoints_2d_list.append(face_keypoints_2d)

        face_keypoints_2d_arr = np.array(face_keypoints_2d_list)

        # save to h5
        group = hf.create_group(f"{identity_id}/{video_id}/{utterance_id}")
        group['expressions'] = expressions
        group['poses'] = poses
        group['betas'] = betas

        group['camera_matrix'] = camera_matrix

        group['face_keypoints_2d'] = face_keypoints_2d_arr

# build scheme
with h5py.File(h5_path, mode='r', libver='latest') as h5f:
    scheme = []
    for identity_id in tqdm(h5f):
        for video_id in h5f[identity_id]:
            for utterance_id in h5f[identity_id][video_id]:
                seq_length = h5f[identity_id][video_id][utterance_id]['expressions'].shape[0]
                for seq_index in range(seq_length):
                    scheme.append((identity_id, video_id, utterance_id, seq_index))

scheme = sorted(scheme)     

with open(scheme_path, 'wb') as f:
    pickle.dump(scheme, f)
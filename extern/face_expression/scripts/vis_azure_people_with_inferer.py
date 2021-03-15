import os
import sys
import shutil
import subprocess
from tqdm.notebook import tqdm
import time

import cv2
import numpy as np

import torch
from torch import nn

from matplotlib import pylab as plt

sys.path.append("..")
import utils.common, utils.vis
from utils.common import Timer

sys.path.append("/Vol0/user/k.iskakov/dev/pykinect")
from pose_dataset_example import *

# input
op_root = "/Vol1/dbstore/datasets/k.iskakov/azure_people_test/openpose_vakhitov_format"
smplx_root = "/Vol1/dbstore/datasets/a.vakhitov/kinect_dataset/full_testcap_30"

# setup dirs for result
result_root_dir = "/Vol1/dbstore/datasets/k.iskakov/projects/face_expression"
result_dir = os.path.join(result_root_dir, "artifacts", "azure_people_video_inferer")

frame_dir = os.path.join(result_dir, "frames")
output_video_path = os.path.join(result_dir, "video.mp4")

shutil.rmtree(output_video_path, ignore_errors=True)
shutil.rmtree(frame_dir, ignore_errors=True)
os.makedirs(frame_dir, exist_ok=True)

# vis
def load_joints_poses_dict(root_folder_bt, pid_lbl, dev_lbl):
    joints_json_path = root_folder_bt + '/' + pid_lbl + '/bt.json'
    with open(joints_json_path) as json_file:
        joints_poses_list = json.load(json_file)

    joints_poses_dict = dict()
    for i in tqdm(range(len(joints_poses_list))):
        try:
            joints_poses, joints_mask = get_kinect_joints(joints_poses_list, i, dev_lbl)
            joints_poses_dict[joints_poses_list[i]['frame_index']] = joints_poses
        except Exception as e:
            pass
            
    return joints_poses_dict

root_folder_bt = "/Vol0/user/r.bashirov/workdir/git/data/smplx_kinect/test_capture/offline_processor2"
root_folder_rgb = "/Vol0/user/r.bashirov/workdir/git/data/smplx_kinect/test_capture/offline_processor2"

pk_root = "/Vol1/dbstore/datasets/a.vakhitov/projects/pykinect_fresh/pykinect/data/fit_realtime_data"
vposer_ckpt = "/Vol1/dbstore/datasets/a.vakhitov/projects/pykinect_fresh/smplify-x/smplify-x-data/vposer_v1_0"

fit_type = 'joints_op_face_hand'

start_index, n_frames, step = 10, 20, 1
frame_ids = np.arange(start_index, start_index + n_frames, step)

fit_dataset_path = smplx_root

dev_lbl = '000062692912'
pid_lbl = '04_simple'

# load device calibs
mkv_meta_json_path = root_folder_bt + '/' + pid_lbl + '/' + dev_lbl + '/mkv_meta.json'
if not os.path.exists(mkv_meta_json_path):
    print('dev not found')
K_rgb, K_d, T_depth_to_rgb = read_device_calibs(mkv_meta_json_path)

joints_poses_dict = load_joints_poses_dict(root_folder_bt, pid_lbl, dev_lbl)

# load inferer
# import lib
FACE_EXPRESSION_SRC_ROOT = "/Vol0/user/k.iskakov/dev/face_expression"
sys.path.append(FACE_EXPRESSION_SRC_ROOT)
from inferer import Inferer

# setup device and checkpoint
device = 'cuda:0'

config_path = "/Vol1/dbstore/datasets/k.iskakov/share/face_expression/gold_checkpoints/siamese+mediapipe_normalization+use_beta-false+checkpoint_000018/config.yaml"
checkpoint_path = "/Vol1/dbstore/datasets/k.iskakov/share/face_expression/gold_checkpoints/siamese+mediapipe_normalization+use_beta-false+checkpoint_000018/checkpoint_000018.pth"

# device = 'cpu'
device = 'cuda:0'
inferer = Inferer(
    config_path,
    checkpoint_path,
    K_rgb,
    T_depth_to_rgb,
    device=device
)

count = 0
times = []
for fid in tqdm(frame_ids):
    print()
    if fid not in joints_poses_dict:
        continue
        
    joints_poses = joints_poses_dict[fid]

    # load rgb image   
    img_path = root_folder_rgb + '/' + pid_lbl + '/' + dev_lbl + '/color_undistorted/' + str(fid).zfill(6) + '.jpg'
    img = cv2.imread(img_path)[:, :, ::-1]

    im_size = (img.shape[1], img.shape[0])

    if img is None:
        print('no rgb frame')
    
    ### (!) prediction
    with Timer('-> inferer.forward'):
        expression_pred, jaw_pose_pred, keypoints_2d, face_bbox = inferer.forward(img, joints_poses, beta=None)

    count += 1
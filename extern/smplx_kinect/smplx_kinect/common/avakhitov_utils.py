import numpy as np
import cv2
import os.path as osp
import json

from human_body_prior.tools.model_loader import load_vposer

import torch

vposer_ckpt = '/Vol1/dbstore/datasets/a.vakhitov/projects/pykinect_fresh/smplify-x/smplify-x-data/vposer_v1_0/'


def load_avakhitov_fits_vposer(vposer, part_path, dev_lbl):
    poses = np.load(part_path + '/poses.npy')[:-1]
    face_expressions = np.load(part_path + '/expressions.npy')[:-1] * 1e2
    betas = np.load(part_path + '/betas.npy')
    fid_lst = np.load(part_path + '/fid_lst.npy')
    with open(part_path + '/config.json', 'r') as f:
        config = json.load(f)
    # do we use vposer embeddings
    is_vposer = config['is_vposer']
    # gender of a subject
    is_male = config['is_male']
    # id of a device (used to decode the rigid pose of the device)

    assert len(fid_lst) == len(poses), f'{len(fid_lst)} != {len(poses)}'
    assert len(fid_lst) == len(face_expressions), f'{len(fid_lst)} != {len(face_expressions)}'
    n = len(poses)

    frame_index2fit_index = {
        fid_lst[i]: i
        for i in range(n)
    }

    # load the device pose
    dev_lst = config['dev_lst']
    dev_id = 0
    while dev_lst[dev_id] != dev_lbl:
        dev_id += 1
    dev_orient = None
    dev_trans = None
    if dev_id > 0:
        dev_orient = np.load(part_path + '/dev_orient.npy')
        dev_trans = np.load(part_path + '/dev_trans.npy')

    rot = poses[:, -3:]
    trans = poses[:, -6:-3]

    if is_vposer:
        pose_body_vp = torch.tensor(poses[:, 0:32])
        # convert from vposer to rotation matrices
        pose_body_list = []
        for i in range(n):
            pose_body_mats = vposer.decode(pose_body_vp[i]).reshape(-1, 3, 3).detach().cpu().numpy()
            pose_body = np.zeros(63)
            for i in range(0, pose_body_mats.shape[0]):
                rot_vec, jac = cv2.Rodrigues(pose_body_mats[i])
                pose_body[3 * i: 3 * i + 3] = rot_vec.reshape(-1)
            pose_body_list.append(pose_body)
        pose_body = np.array(pose_body_list)
        pose_jaw = poses[:, 32:35]
        pose_eye = poses[:, 35:41]
        pose_hand = poses[:, 41:-6]
    else:
        pose_body = poses[:, 0:63]
        pose_jaw = poses[:, 63:66]
        pose_eye = poses[:, 66:72]
        pose_hand = poses[:, 72:-6]

    if dev_orient is not None:
        for i in range(n):
            rot_mat = cv2.Rodrigues(rot[i].reshape(3, 1))[0]
            dev_mat = cv2.Rodrigues(dev_orient.reshape(3, 1))[0]
            rot_mat = dev_mat @ rot_mat
            rot[i] = cv2.Rodrigues(rot_mat)[0].reshape(-1)
            trans[i] = (dev_mat @ trans[i].reshape(3, 1) + dev_trans.reshape(3, 1)).reshape(-1)
    result = {
        'global_rvec': rot,
        'global_tvec': trans,
        'body_pose': pose_body,
        'hand_pose': pose_hand,
        'jaw_pose': pose_jaw,
        'eye_pose': pose_eye,
        'face_expression': face_expressions,
        'betas': betas,
        'n': n,
        'frame_index2fit_index': frame_index2fit_index,
        'is_male': is_male,
        'is_vposer': is_vposer
    }
    return result


def load_avakhitov_fits(dp, load_betas=True, load_body_poses=True, load_expressions=False, load_fid_lst=True):
    result = dict()
    for flag, k, fn_no_ext in [
        [load_betas, 'betas', 'betas'],
        [load_body_poses, 'body_poses', 'poses'],
        [load_expressions, 'expressions', 'expressions'],
        [load_fid_lst, 'fid_lst', 'fid_lst']
    ]:
        if flag:
            load_fp = osp.join(dp, f'{fn_no_ext}.npy')
            try:
                loaded = np.load(load_fp)
            except:
                print(load_fp)
                raise Exception()

            if fn_no_ext == 'poses':
                #load the vposer model
                if loaded.shape[1] == 69:
                    pose_body = loaded[:, 0:32]
                else:
                    vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
                    vposer.eval()
                    pose_body_vp = torch.tensor(loaded[:, 0:32])      
                    #convert from vposer to rotation matrices
                    pose_body_mats = vposer.decode(pose_body_vp).reshape(len(loaded), -1, 3, 3).detach().cpu().numpy()
                    pose_body = np.zeros((pose_body_mats.shape[0], 63))
                    for i in range(0, pose_body_mats.shape[0]):
                        for j in range(0, pose_body_mats.shape[1]):
                            rot_vec, jac = cv2.Rodrigues(pose_body_mats[i,j])
                            pose_body[i, 3*j : 3*j+3] = rot_vec.reshape(-1)   
                result[k] = pose_body
                result['global_rvecs'] = loaded[:, -3:]
                result['global_tvecs'] = loaded[:, -6:-3]
                result['n'] = len(loaded)
            else:
                result[k] = loaded
    return result


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def get_selected_ids(id_sel_set, req_ids):
    ss_sort = np.argsort(id_sel_set)
    req_sort = np.argsort(req_ids)
    id_ss_srt = id_sel_set[ss_sort]
    id_ss_pos = np.arange(0, len(id_sel_set))[ss_sort]
    req_srt = req_ids[req_sort]
    req_srt_pos = -1 * np.ones(len(req_srt), dtype=int)
    i = 0
    j = 0
    while i < len(id_ss_srt) and j < len(req_srt):
        if req_srt[j] == id_ss_srt[i]:
            req_srt_pos[j] = id_ss_pos[i]
            i += 1
            j += 1
        elif req_srt[j] < id_ss_srt[i]:
            j += 1
        elif id_ss_srt[i] < req_srt[j]:
            i += 1
    req_ids_ans = -1 * np.ones(len(req_srt), dtype=int)
    req_ids_ans[req_sort] = req_srt_pos
    return req_ids_ans



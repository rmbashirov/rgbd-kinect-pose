import os.path as osp
import numpy as np
from scipy.special import softmax

from .smplx_model import ExpBodyModel


def load_patched_smplx(pykinect_data_dp, gender, num_hand_pca, device, is_w_add=True, is_s2k=True):
    if gender == 'male':
        bm_path = osp.join(pykinect_data_dp, 'body_models/smplx/SMPLX_MALE.npz')
        s2k_path = osp.join(pykinect_data_dp, 'rob75_val/s2k_m.npy')
    elif gender == 'female':
        bm_path = osp.join(pykinect_data_dp, 'body_models/smplx/SMPLX_FEMALE.npz')
        s2k_path = osp.join(pykinect_data_dp, 'rob75_val/s2k_f.npy')
    else:
        raise Exception(f'gender {gender} unknown')

    if is_s2k:
        s2k = np.load(s2k_path)
    else:
        s2k = None

    if is_w_add:
        kinect_vert_weights_path = osp.join(pykinect_data_dp, 'rob75_val/weights.npy')
        w_add = np.load(kinect_vert_weights_path)
        w_add = softmax(w_add, axis=1)
    else:
        w_add = None

    smplx_model = ExpBodyModel(
        bm_path,
        is_hand_pca=True,
        num_hand_pca=num_hand_pca,
        fe_scale=1e2,
        s2v=s2k,
        w_add=w_add,
        comp_device=device
    )

    # if is_J:
    #     J_path = osp.join(pykinect_data_dp, 'rob75_val/J.npy')
    #     J = np.load(J_path)
    # else:
    #     J = None

    return smplx_model

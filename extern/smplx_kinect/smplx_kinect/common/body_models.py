import smplx
import torch
import numpy as np
import os
import os.path as osp


def tt(a, device, unsqueeze=True):
    result = torch.tensor(a, dtype=torch.float32, device=device)
    if unsqueeze:
        result = result.unsqueeze(0)
    return result


class BodyModelsWrapper:
    def __init__(self):
        pass

    def tt(self, a, **kwargs):
        if a is None:
            return None
        else:
            return tt(a, device=self.device, **kwargs)

    def get_output(self, **kwargs):
        raise NotImplementedError

    def get_vertices(self, **kwargs):
        """
        :param kwargs: see get_output:
        :return: in case of smpl: (6890, 3)
        :return: in case of smplx: (10475, 3)
        """
        output = self.get_output(**kwargs)
        return output.vertices.detach()[0].contiguous()


class DefaultBodyModelsWrapper(BodyModelsWrapper):
    def __init__(self, model_type, body_models_dp, device):
        super().__init__()
        self.device = device
        self.models = dict()
        for gender in ['female', 'male']:
            model = smplx.create(
                body_models_dp,
                model_type=model_type,
                gender=gender,
                batch_size=1,
                create_transl=False
            ).to(device=self.device)
            model.eval()
            self.models[gender] = model


class SMPLWrapper(DefaultBodyModelsWrapper):
    def __init__(self, body_models_dp, device):
        super().__init__('smpl', body_models_dp, device)

    def get_output(self, gender, betas, body_pose, rvec, tvec):
        """
        :param gender: 'female' or 'male'
        :param betas: body shape numpy array of shape (10,)
        :param body_pose: numpy array of shape (23*3,)
        :param rvec: global rotation numpy array of shape (3,)
        :param tvec: global translation numpy array of shape (3,)
        :return: model output
        """
        return self.models[gender](
            betas=self.tt(betas),
            body_pose=self.tt(body_pose),
            global_orient=self.tt(rvec),
            transl=self.tt(tvec),
            return_verts=True,
            pose2rot=True
        )


class SMPLXWrapper(DefaultBodyModelsWrapper):
    def __init__(self, body_models_dp, device):
        super().__init__('smplx', body_models_dp, device)

    def get_output(
        self,
        gender, betas, body_pose, rvec, tvec,
        left_hand_pose=None, right_hand_pose=None,
        expression=None, jaw_pose=None, leye_pose=None, reye_pose=None
    ):
        """
        :param gender: 'female' or 'male'
        :param betas: body shape numpy array of shape (10,)
        :param body_pose: numpy array of shape (21*3,)
        :param rvec: global rotation numpy array of shape (3,)
        :param tvec: global translation numpy array of shape (3,)
        :return: model output
        """
        return self.models[gender](
            betas=self.tt(betas),
            body_pose=self.tt(body_pose),
            global_orient=self.tt(rvec),
            transl=self.tt(tvec),
            return_verts=True,
            pose2rot=True
        )


class HandsSMPLXWrapper(BodyModelsWrapper):
    def __init__(self,
         body_models_dp,
         use_pca=True, num_pca_comps=45, flat_hand_mean=False,
         device=torch.device('cpu')
    ):
        super().__init__()
        self.models = dict()
        self.device = device
        for gender in ['female', 'male']:
            model = smplx.create(
                body_models_dp,
                model_type='smplx',
                gender=gender,
                batch_size=1,
                use_pca=use_pca,
                num_pca_comps=num_pca_comps,
                flat_hand_mean=flat_hand_mean,
                create_transl=False
            ).to(device=self.device)
            model.eval()
            self.models[gender] = model

    def hand_pose_pca2aa(self, pca, gender, is_left, add_mean=False, substract_mean=False):
        model = self.models[gender]
        hand_components = model.left_hand_components if is_left else model.right_hand_components
        hand_pose = torch.einsum('bi,ij->bj', [pca, hand_components])
        if add_mean or substract_mean:
            pose_mean = model.left_hand_mean if is_left else model.right_hand_mean
            if add_mean:
                hand_pose += pose_mean
            else:
                hand_pose -= pose_mean
        return hand_pose

    def get_output(
        self,
        gender, betas, body_pose, rvec, tvec,
        left_hand_pose=None, right_hand_pose=None,
        expression=None, jaw_pose=None, leye_pose=None, reye_pose=None
    ):
        return self.models[gender](
            betas=self.tt(betas),
            global_orient=self.tt(rvec),
            body_pose=self.tt(body_pose),
            left_hand_pose=self.tt(left_hand_pose),
            right_hand_pose=self.tt(right_hand_pose),
            transl=self.tt(tvec),
            expression=self.tt(expression),
            jaw_pose=self.tt(jaw_pose),
            return_verts=True,
            pose2rot=True
        )
import torch
from torch import nn

from face_expression.utils.angle_representation import (
    universe_convert
)


class MAECriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


class MSECriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return torch.mean((pred - target) ** 2)


class KeypointL2Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, keypoints_pred, keypoints_target):
        return torch.mean(torch.sqrt(torch.sum((keypoints_pred - keypoints_target) ** 2, dim=2)))


class JawPoseCriterion(torch.nn.Module):
    def __init__(self, criterion_type='MAE', rotmat_representation=True):
        super().__init__()

        self.criterion = {
            'MAE': MAECriterion,
            'MSE': MSECriterion
        }[criterion_type]()

        self.rotmat_representation = rotmat_representation

    def forward(self, jaw_pose_pred, jaw_pose_target):
        if self.rotmat_representation:
            jaw_pose_pred = universe_convert(jaw_pose_pred, 'aa', 'rotmtx')
            jaw_pose_target = universe_convert(jaw_pose_target, 'aa', 'rotmtx')

        loss = self.criterion(jaw_pose_pred, jaw_pose_target)

        return loss

import collections
import pydoc

import torch
import torch.nn.functional as F

from face_expression import utils


def infer_smplx(smplx_model, expression, pose, beta):
    batch_size = expression.shape[0]
    device = expression.device
    
    # extract
    jaw_pose = pose[:, 63:66]
    translation = pose[:, 84:87]
    root_orientation = pose[:, 87:90]

    eye_pose = torch.zeros(batch_size, 6).to(device)
    body_pose = torch.zeros(batch_size, 63).to(device)
    hand_pose = torch.zeros(batch_size, 12).to(device)
    
    # infer SMPLX model
    keypoints_3d, rotation_matrices, verts = smplx_model(
        root_orient=root_orientation,
        pose_body=body_pose,
        pose_hand=hand_pose,
        pose_jaw=jaw_pose,
        pose_eye=eye_pose,
        betas=beta,
        trans=translation,
        expression=expression
    )

    return keypoints_3d, rotation_matrices, verts


def project_keypoints_3d(keypoints_3d, projection_matrix):
    keypoints_3d_homo = F.pad(keypoints_3d, pad=[0, 1], mode='constant', value=0.0)

    keypoints_2d_homo_proj = torch.bmm(keypoints_3d_homo, projection_matrix.transpose(1, 2))
    keypoints_2d_proj = keypoints_2d_homo_proj[:, :, :2] / keypoints_2d_homo_proj[:, :, 2:]

    return keypoints_2d_proj


def infer_smplx_keypoints_2d(smplx_model, expression, pose, beta, projection_matrix):
    keypoints_3d, rotation_matrices, verts = infer_smplx(smplx_model, expression, pose, beta)
    keypoints_2d = project_keypoints_3d(keypoints_3d, projection_matrix)

    return keypoints_2d


def get_dataloaders(config, splits=('train', 'val')):
    dataloaders = collections.OrderedDict()
    
    for dataset_type in splits:
        data_config = config.data[dataset_type]

        dataset_cls = pydoc.locate(data_config.dataset.cls)
        dataset = dataset_cls(**data_config.dataset.args)

        dataloader_args = data_config.dataloader.args
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=dataloader_args.batch_size,
            num_workers=dataloader_args.num_workers,
            sampler=utils.common.get_data_sampler(dataset, shuffle=dataloader_args.shuffle, is_distributed=False),  ## TODO: check distributed
            pin_memory=True,
            drop_last=dataloader_args.drop_last
        )

        dataloaders[dataset_type] = dataloader

    return dataloaders


def get_logger(config):
    logger_cls = pydoc.locate(config.log.logger.cls)
    logger = logger_cls(config, config.log.project_dir, config.log.project_name, config.log.experiment_name)

    return logger

import os
import pydoc
from omegaconf import OmegaConf

import numpy as np
import torch

from face_expression.third_party.face_mesh_mediapipe import FaceMeshMediaPipe


class Inferer:
    KINECT_FACE_INDICES = [27, 28, 29, 30, 31]

    def __init__(
            self,
            config_path,
            checkpoint_path,
            device='cuda:0'
        ):
        with open(config_path) as f:
            self.config = OmegaConf.load(f)

        self.device = device
            
        # load predictor
        predictor_cls = pydoc.locate(self.config.model.predictor.cls)

        predictor_args = {} if self.config.model.predictor.args is None else self.config.model.predictor.args
        self.predictor = predictor_cls(**predictor_args)

        self.use_keypoints_3d = predictor_args.use_keypoints_3d
        
        ## load weights
        state_dict = torch.load(checkpoint_path)
        self.predictor.load_state_dict(state_dict['predictor'])
        self.predictor.to(device)
        self.predictor.eval()
        
        # load face mesh model
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "face_mesh_mediapipe", "models")
        anchors_path = os.path.join(models_dir, "face_anchors.csv")
        detection_model_path = os.path.join(models_dir, "face_detection_front.tflite")
        landmark_model_path = os.path.join(models_dir, "face_landmark.tflite")

        self.face_mesh_model = FaceMeshMediaPipe(anchors_path, detection_model_path, landmark_model_path, bbox_scale=1.5)

    def forward(self, image_cropped, beta=None):
        """Predicts expression and jaw pose from crom image of head (and maybe beta)
        Args:
            image_cropped (numpy array of shape (height, width, 3), RGB, uint8): cropped image of head
            beta (numpy array of shape (10,), float32): Beta (shape parameters) of human
        Returns:
            expression (numpy array of shape (10,), float32): Expression
            jaw_pose (numpy array of shape (3,), float32): Jaw pose
            keypoints_2d (numpy array of shape (J, 2), float32): 2D keypoints
        """
        keypoints_3d, keypoints_3d_normed = self.face_mesh_model(image_cropped)

        if keypoints_3d is None:
            expression = np.zeros(10)
            jaw_pose = np.zeros(3)
            keypoints_2d = np.zeros(468, 2)
            return expression, jaw_pose, keypoints_2d
        
        if self.use_keypoints_3d:
            keypoints, keypoints_normed = keypoints_3d, keypoints_3d_normed
        else:
            keypoints, keypoints_normed = keypoints_3d[:, :2], keypoints_3d_normed[:, :2]

        keypoints_normed_t = torch.from_numpy(keypoints_normed).unsqueeze(0).type(torch.float32).to(self.device)
        if beta is not None:
            beta = torch.from_numpy(beta).unsqueeze(0).type(torch.float32).to(self.device)

        expression_pred, jaw_pose_pred = self.predictor.forward(keypoints_normed_t.contiguous(), beta)

        expression_pred = expression_pred[0].detach().cpu().numpy()
        jaw_pose_pred = jaw_pose_pred[0].detach().cpu().numpy()

        return expression_pred, jaw_pose_pred, keypoints_3d[:, :2]

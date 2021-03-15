import numpy as np
import cv2

from face_expression import utils


class Cropper:
    KINECT_FACE_INDICES = [27, 28, 29, 30, 31]

    def __init__(self, rvec, tvec, K, dist=None):
        self.rvec = rvec
        self.tvec = tvec
        self.K = K
        self.dist = np.zeros(8, dtype=np.float32)
        if dist is not None:
            self.dist[:len(dist)] = dist
        self.dist = dist

    def forward(self, image, kinect_joints_3d):
        """

        Args:
            image (numpy array of shape (height, width, 3), RGB, uint8): Input image of head
            kinect_joints_3d (numpy array of shape (32, 3), RGB, float32): Kinect 3D joints

        Returns:
            image_cropped: (numpy array of shape (height, width, 3), RGB, uint8): cropped image of head
            face_bbox (numpy array of shape (4,), float32): Face bbox of format (left, top, right, down)
        """
        kinect_joints_2d, _ = cv2.projectPoints(kinect_joints_3d, self.rvec, self.tvec, self.K, self.dist)
        kinect_joints_2d = kinect_joints_2d[:, 0, :]
        face_bbox = self.crop_bbox_by_keypoints_2d(kinect_joints_2d[self.KINECT_FACE_INDICES], scale=2)
        image_cropped = utils.common.crop_image(image, face_bbox)

        return image_cropped, face_bbox

    @staticmethod
    def crop_bbox_by_keypoints_2d(keypoints_2d, scale=2):
        x_min, y_min = np.min(keypoints_2d, axis=0)
        x_max, y_max = np.max(keypoints_2d, axis=0)

        width = x_max - x_min
        height = y_max - y_min

        # size = max(width, height)
        bbox = x_min, y_min, x_min + width, y_min + height
        bbox = utils.common.get_square_bbox(bbox)

        bbox = utils.common.scale_bbox(bbox, scale)

        return bbox

    @staticmethod
    def project_kinect_joints(kinect_joints_3d, P_cw, im_size):
        n_j = kinect_joints_3d.shape[0]
        j_h = np.concatenate([kinect_joints_3d, np.ones((n_j, 1))], axis=1)
        j_cam_proj = j_h @ P_cw.T
        j_scr = j_cam_proj[:, 0:2] / np.tile(j_cam_proj[:, 2].reshape(-1, 1), (1, 2))
        j_scr_i = (j_scr + 0.5).astype(int)
        j_scr_i[:, 0] = np.clip(j_scr_i[:, 0], 0, im_size[0] - 1)
        j_scr_i[:, 1] = np.clip(j_scr_i[:, 1], 0, im_size[1] - 1)

        return j_scr_i

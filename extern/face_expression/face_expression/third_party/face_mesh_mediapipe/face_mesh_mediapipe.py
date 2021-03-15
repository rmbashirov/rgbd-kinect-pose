import csv
import numpy as np
import cv2

import tensorflow as tf

np.seterr(over='ignore')


class FaceMeshMediaPipe:
    # no NVIDIA GPU support for TFLite models: https://github.com/tensorflow/tensorflow/issues/40706
    
    def __init__(self, anchors_path, detection_model_path, landmark_model_path, bbox_scale=1.5):
        self.bbox_scale = bbox_scale
        
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        self.detector_model = tf.lite.Interpreter(detection_model_path)
        self.detector_model.allocate_tensors()
        
        self.landmark_model = tf.lite.Interpreter(landmark_model_path)
        self.landmark_model.allocate_tensors()
        
        # detector
        detector_input_details = self.detector_model.get_input_details()
        detector_output_details = self.detector_model.get_output_details()
        
        self.detector_in_idx = detector_input_details[0]['index']
        self.detector_out_reg_idx = detector_output_details[0]['index']
        self.detector_out_clf_idx = detector_output_details[1]['index']
        
        self.detector_input_shape = detector_input_details[0]['shape'][1:3]
        
        # landmark
        landmark_input_details = self.landmark_model.get_input_details()
        landmark_output_details = self.landmark_model.get_output_details()
        
        self.landmark_in_idx = landmark_input_details[0]['index']
        self.landmark_out_reg_idx = landmark_output_details[0]['index']
        
        self.landmark_input_shape = landmark_input_details[0]['shape'][1:3]


    def preprocess_image(self, image):
        # pad
        shape = np.array(image.shape[:2])
        max_side_length = shape.max()
        pad = (max_side_length - shape[:2]).astype('uint32') // 2
        
        image_pad = np.pad(
            image,
            ((pad[0], max_side_length - shape[0] - pad[0]), (pad[1], max_side_length - shape[1] - pad[1]), (0, 0)),
            mode='constant'
        )
        
        # resize
        image_pad = cv2.resize(image_pad, tuple(self.detector_input_shape), interpolation=cv2.INTER_AREA)
        
        # norm
        image_norm = 2 * (image_pad/255.0 - 0.5)
        image_norm = image_norm.astype('float32')

        return image_pad, image_norm, pad


    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x) )

    def detect_face(self, image_norm):
        assert -1 <= image_norm.min() and image_norm.max() <= 1, "image_norm must be in range [-1, 1]"
        assert image_norm.shape == (*self.detector_input_shape, 3), "image_norm shape must be (128, 128, 3)"

        # predict face location and initial landmarks
        image_norm = np.expand_dims(image_norm, axis=0)
        self.detector_model.set_tensor(self.detector_in_idx, image_norm)
        self.detector_model.invoke()

        out_reg = self.detector_model.get_tensor(self.detector_out_reg_idx)[0]
        out_clf = self.detector_model.get_tensor(self.detector_out_clf_idx)[0, :, 0]

        # find the best prediction
        # TODO: replace it with non-max suppression
        detection_mask = self._sigm(out_clf) > 0.3
        candidate_detections = out_reg[detection_mask]
        candidate_anchors = self.anchors[detection_mask]

        if candidate_detections.shape[0] == 0:
            return None, None, None

        # picking the widest suggestion while NMS is not implemented
        max_idx = np.argmax(candidate_detections[:, 3])
        center_wo_offset = candidate_anchors[max_idx, :2] * 128

        keypoints = candidate_detections[max_idx, 4:].reshape(-1, 2) + center_wo_offset.reshape(1, 2)

        x_0, y_0 = keypoints[0]
        x_1, y_1 = keypoints[1]
        theta = -np.arctan2(-(y_1 - y_0), x_1 - x_0)

        bbox_cx, bbox_cy, bbox_w, bbox_h = candidate_detections[max_idx, :4]
        bbox_center, bbox_size = np.array([bbox_cx + center_wo_offset[0], bbox_cy + center_wo_offset[1]]), np.array([bbox_w, bbox_h])

        return bbox_center, bbox_size, theta
    
    def predict_landmarks(self, image_norm_crop):
        assert -1 <= image_norm_crop.min() and image_norm_crop.max() <= 1, "image_norm must be in range [-1, 1]"
        assert image_norm_crop.shape == (*self.landmark_input_shape, 3), "image_norm shape must be (192, 192, 3)"

        image_norm_crop = np.expand_dims(image_norm_crop, axis=0)
        self.landmark_model.set_tensor(self.landmark_in_idx, image_norm_crop)
        self.landmark_model.invoke()

        keypoints_3d = self.landmark_model.get_tensor(self.landmark_out_reg_idx)[0].reshape(-1, 3)

        keypoints_3d[:, 2] = keypoints_3d[:, 2] / image_norm_crop.shape[1]

        return keypoints_3d
        
    
    def crop_with_rotation(self, image_norm, bbox_center, bbox_size, theta):
        lu = (bbox_center - bbox_size / 2).astype(int)
        rb = (bbox_center + bbox_size / 2).astype(int)

        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])

        bbox_w, bbox_h = bbox_size
        lu = bbox_center + np.array([-bbox_w/2, -bbox_h/2])
        lb = bbox_center + np.array([-bbox_w/2, +bbox_h/2])
        rb = bbox_center + np.array([+bbox_w/2, +bbox_h/2])
        ru = bbox_center + np.array([+bbox_w/2, -bbox_h/2])

        bbox_pts = np.stack([lu, lb, rb, ru], axis=0)
        bbox_pts_rotated = (bbox_pts - bbox_center) @ rotation_matrix.T + bbox_center

        target_pts = np.array([
            [0.0, 0.0],
            [0.0, self.landmark_input_shape[1]],
            [self.landmark_input_shape[0], self.landmark_input_shape[1]],
            [self.landmark_input_shape[0], 0.0]
        ])

        transformation_matrix = cv2.getPerspectiveTransform(
            bbox_pts_rotated.astype(np.float32), target_pts.astype(np.float32)
        )
        
        image_norm_crop = cv2.warpPerspective(
            image_norm, transformation_matrix, tuple(self.landmark_input_shape)
        )
        
        return image_norm_crop, transformation_matrix


    def __call__(self, image):
        orig_image_shape = image.shape[:2]
        image_pad, image_norm, pad = self.preprocess_image(image)

        bbox_center, bbox_size, theta = self.detect_face(image_norm)
        if bbox_center is None:
            return None, None
        
        bbox_size *= self.bbox_scale

        image_norm_crop, transformation_matrix = self.crop_with_rotation(image_norm, bbox_center, bbox_size, theta)
        
        keypoints_3d_orig = self.predict_landmarks(image_norm_crop)
    
        # transform back to image space
        keypoints_2d = keypoints_3d_orig[:, :2].copy()

        transformation_matrix_inv = np.linalg.inv(transformation_matrix)
        keypoints_2d = cv2.perspectiveTransform(keypoints_2d[None, :, :], transformation_matrix_inv)[0]

        keypoints_2d *= max(orig_image_shape) / max(image_pad.shape)
        keypoints_2d -= np.array([pad[1], pad[0]])

        # return back depth
        keypoints_3d = np.concatenate([keypoints_2d, keypoints_3d_orig[:, 2:]], axis=1)

        # norm keypoints 3d norm
        image_norm_crop_shape_half = np.array([image_norm_crop.shape[1], image_norm_crop.shape[0]]) / 2
        keypoints_3d_orig[:, :2] = (keypoints_3d_orig[:, :2] - image_norm_crop_shape_half) / image_norm_crop_shape_half
        keypoints_3d_normed = keypoints_3d_orig
        
        return keypoints_3d, keypoints_3d_normed

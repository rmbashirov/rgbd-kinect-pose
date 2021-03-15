import os
import sys
import json
import pickle
import h5py
from tqdm import tqdm

import numpy as np
import torch
import cv2
import scipy.spatial

import hydra

from face_expression import utils
from face_expression.third_party.face_mesh_mediapipe import FaceMeshMediaPipe


# class VoxCeleb2FaceDataset(torch.utils.data.Dataset):    
#     def __init__(
#             self,
#             h5_path,
#             scheme_path,
#             image_root,
#             return_images=True,
#             bbox_scale=2.0,
#             image_shape=(256, 256),
#             sample_range=None
#         ):
        
#         self.h5_path = h5_path
#         self.scheme_path = scheme_path
        
#         self.image_root = image_root
#         self.return_images = return_images
#         self.bbox_scale = bbox_scale
#         self.image_shape = image_shape

#         self.sample_range = sample_range

#         # load scheme
#         with open(scheme_path, 'rb') as f:
#             self.scheme = pickle.load(f)
        
#         if sample_range is not None:
#             self.scheme = [self.scheme[i] for i in range(sample_range[0], sample_range[1], sample_range[2])]
        
#     def open_h5_file(self):
#         self.h5f = h5py.File(self.h5_path, mode='r')
    
#     def load_image(self, identity_id, video_id, utterance_id, seq_index):
#         image_dir = os.path.join(self.image_root, identity_id, video_id, utterance_id)
        
#         names = sorted(os.listdir(image_dir))
#         if seq_index < len(names):
#             name = names[seq_index]
#             path = os.path.join(image_dir, name)
#             image = cv2.imread(path)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         else:
#             # black image mock
#             name = names[0]
#             path = os.path.join(image_dir, name)
#             image = cv2.imread(path)
#             image = np.zeros(image.shape, dtype=np.uint8)

#         return image
    
#     def get_camera_matrix(self, h, w):
#         fx, fy = 3000.0, 3000.0
#         cx, cy = w/2, h/2
#         camera_martix = np.array([
#             [fx, 0.0, cx],
#             [0.0, fy, cy],
#             [0.0, 0.0, 1.0]
#         ])
        
#         return camera_martix
    
#     def get_transformation_matrix(self):
#         transformation_matrix = np.eye(3, 4)
        
#         return transformation_matrix
    

#     def get_bbox(self, keypoints_2d):
#         left, top, right, down = (
#             keypoints_2d[:, 0].min(),
#             keypoints_2d[:, 1].min(),
#             keypoints_2d[:, 0].max(),
#             keypoints_2d[:, 1].max()
#         )
        
#         # convex_hull = scipy.spatial.ConvexHull(points)
#         # center_x, center_y = (np.mean(convex_hull.points[convex_hull.vertices, axis]) for axis in (0, 1))
        
#         center_x, center_y = (left + right) / 2, (top + down) / 2
#         w, h = right - left, down - top
#         bbox = (
#             center_x - w/2,
#             center_y - h/2,
#             center_x + w/2,
#             center_y + h/2
#         )

#         bbox = utils.common.utils.common.get_square_bbox(bbox)
#         bbox = utils.common.utils.common.scale_bbox(bbox, self.bbox_scale)
        
#         return bbox

#     def normalize_keypoints_2d(self, keypoints_2d):
#         convex_hull = scipy.spatial.ConvexHull(keypoints_2d)
#         center = np.mean(convex_hull.points[convex_hull.vertices], axis=0)

#         keypoints_2d = (keypoints_2d - center) / np.sqrt(convex_hull.area)

#         return keypoints_2d

#     def load_sample(self, identity_id, video_id, utterance_id, seq_index):
#         sample = dict()
        
#         # load h5_data
#         try:
#             h5_data = self.h5f[identity_id][video_id][utterance_id]
#         except Exception as e:
#             print(identity_id, video_id, utterance_id, seq_index)
#             print(e)
        
#         sample['expression'] = h5_data['expressions'][seq_index]
#         sample['pose'] = h5_data['poses'][seq_index]
#         sample['beta'] = h5_data['betas'][:]
        
#         sample['keypoints_2d'] = h5_data['face_keypoints_2d'][seq_index]
        
#         # load image
#         if self.return_images:
#             image = self.load_image(identity_id, video_id, utterance_id, seq_index)
#             orig_h, orig_w = image.shape[:2]
            
#             # crop
#             bbox = self.get_bbox(sample['keypoints_2d'])
#             image = utils.common.utils.common.crop_image(image, bbox)
            
#             # resize
#             image = utils.common.utils.common.resize_image(image, self.image_shape)
            
#             image = image / 255.0
#             image = image.transpose(2, 0, 1)
            
#             sample['image'] = image
            
#             # load projection matrix
#             h, w = image.shape[1:3]
#             bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            
#             if 'camera_matrix' in h5_data:
#                 print('hey')
#                 camera_matrix = h5_data['camera_matrix'][:]
#             else:
#                 camera_matrix = self.get_camera_matrix(orig_h, orig_w)

#             camera_matrix = utils.common.utils.common.update_after_crop_and_resize(
#                 camera_matrix, bbox, (w/bbox_w, h/bbox_h)
#             )

#             # update keypoints 2d ufter crop and resize
#             sample['keypoints_2d'][:, 0] -= bbox[0]
#             sample['keypoints_2d'][:, 1] -= bbox[1]
            
#             sample['keypoints_2d'][:, 0] *= w/bbox_w
#             sample['keypoints_2d'][:, 1] *= h/bbox_h
#         else:
#             image = np.zeros((*self.image_shape, 3), dtype=np.uint8)
#             image = image / 255.0
#             image = image.transpose(2, 0, 1)
#             h, w = image.shape[1:3]
#             sample['image'] = image

#             if 'camera_matrix' in h5_data:
#                 camera_matrix = h5_data['camera_matrix'][:]
#             else:
#                 camera_matrix = self.get_camera_matrix(*self.image_shape)
        
#         transformation_matrix = self.get_transformation_matrix()
        
#         projection_matrix = camera_matrix @ transformation_matrix
        
#         sample['camera_matrix'] = camera_matrix
#         sample['projection_matrix'] = projection_matrix
#         sample['h'] = h
#         sample['w'] = w

#         # normalize keypoints 2d
#         sample['keypoints_2d'] = self.normalize_keypoints_2d(sample['keypoints_2d'])
        
#         return sample
    
#     def __len__(self):
#         return len(self.scheme)
    
#     def __getitem__(self, index):
#         # this should be normally done in __init__, but due to DataLoader behaviour 
#         # when num_workers > 1, the h5 file is opened during first data access:
#         # https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983

#         if not hasattr(self, 'h5f'):
#             self.open_h5_file()

#         sample_key = self.scheme[index]
#         sample = self.load_sample(*sample_key)
#         return sample
    
#     @staticmethod
#     def build_scheme(h5f):
#         scheme = []
#         for identity_id in tqdm(h5f):
#             for video_id in h5f[identity_id]:
#                 for utterance_id in h5f[identity_id][video_id]:
#                     seq_length = h5f[identity_id][video_id][utterance_id]['expressions'].shape[0]
#                     for seq_index in range(seq_length):
#                         scheme.append((identity_id, video_id, utterance_id, seq_index))
                    
#         scheme = sorted(scheme)     
#         return scheme
    
#     @staticmethod
#     def preprocess_dataset(face_root, image_root, openpose_root, h5_path):
#         # load scheme
#         scheme = []

#         identity_id_list = sorted(os.listdir(face_root))
#         for identity_id in tqdm(identity_id_list):
#             identity_dir = os.path.join(face_root, identity_id)
#             video_id_list = sorted(os.listdir(identity_dir))

#             for video_id in video_id_list:
#                 video_dir = os.path.join(identity_dir, video_id)
#                 utterance_id_list = sorted(os.listdir(video_dir))

#                 for utterance_id in utterance_id_list:
#                     utterance_dir = os.path.join(video_dir, utterance_id)

#                     scheme.append((identity_id, video_id, utterance_id))

#         scheme = sorted(scheme)

#         # build h5 file
#         with h5py.File(h5_path, 'w') as hf:
#             for (identity_id, video_id, utterance_id) in tqdm(scheme):
#                 # load face
#                 face_dir = os.path.join(face_root, identity_id, video_id, utterance_id, 'joints_op_face')

#                 expressions = np.load(os.path.join(face_dir, 'expressions.npy')) * 100
#                 poses = np.load(os.path.join(face_dir, 'poses.npy'))
#                 betas = np.load(os.path.join(face_dir, 'betas.npy'))

#                 # load openpose keypoints 2d
#                 openpose_dir = os.path.join(openpose_root, identity_id, video_id, utterance_id)

#                 face_keypoints_2d_list = []
#                 names = sorted(os.listdir(openpose_dir))
#                 for name in names:
#                     path = os.path.join(openpose_dir, name)

#                     with open(path) as f:
#                         openpose_data = json.load(f)

#                     face_keypoints_2d = openpose_data['people'][0]['face_keypoints_2d']
#                     face_keypoints_2d = np.array(face_keypoints_2d).reshape(70, 3)
#                     face_keypoints_2d = face_keypoints_2d[:, :2]  # remove confidences

#                     face_keypoints_2d_list.append(face_keypoints_2d)

#                 face_keypoints_2d_arr = np.array(face_keypoints_2d_list)

#                 # save to h5
#                 group = hf.create_group(f"{identity_id}/{video_id}/{utterance_id}")
#                 group['expressions'] = expressions
#                 group['poses'] = poses
#                 group['betas'] = betas

#                 group['face_keypoints_2d'] = face_keypoints_2d_arr


class VoxCeleb2MediapipeDataset(torch.utils.data.Dataset):    
    def __init__(
            self, *,
            h5_path='', scheme_path='',
            image_root='',
            return_keypoints_3d=False,
            return_images=True, bbox_scale=2.0, image_shape=(256, 256),
            sample_range=None
        ):
        assert return_images
        
        self.h5_path = h5_path
        self.scheme_path = scheme_path

        self.return_keypoints_3d = return_keypoints_3d
        
        self.image_root = image_root
        self.return_images = return_images
        self.bbox_scale = bbox_scale
        self.image_shape = image_shape

        self.sample_range = sample_range
        
        # load facemesh model
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "face_mesh_mediapipe", "models")
        anchors_path = os.path.join(models_dir, "face_anchors.csv")
        detection_model_path = os.path.join(models_dir, "face_detection_front.tflite")
        landmark_model_path = os.path.join(models_dir, "face_landmark.tflite")

        self.face_mesh_model = FaceMeshMediaPipe(anchors_path, detection_model_path, landmark_model_path, bbox_scale=1.5)

        # load scheme
        with open(scheme_path, 'rb') as f:
            self.scheme = pickle.load(f)

        if sample_range is not None:
            start = max(0, sample_range[0])
            end = min(len(self.scheme), sample_range[1])
            step = sample_range[2]
            self.scheme = [self.scheme[i] for i in range(start, end, step)]
        
    def open_h5_file(self):
        self.h5f = h5py.File(self.h5_path, mode='r')
    
    def load_image(self, identity_id, video_id, utterance_id, seq_index):
        image_dir = os.path.join(self.image_root, identity_id, video_id, utterance_id)
        if not os.path.exists(image_dir):
            image_dir = os.path.join(self.image_root, identity_id, video_id, 'color_undistorted')
        
        names = sorted(os.listdir(image_dir))
        if seq_index < len(names):
            name = names[seq_index]
            path = os.path.join(image_dir, name)
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # black image mock
            name = names[0]
            path = os.path.join(image_dir, name)
            image = cv2.imread(path)
            image = np.zeros(image.shape, dtype=np.uint8)

        return image
    
    def get_camera_matrix(self, h, w):
        fx, fy = 3000.0, 3000.0
        cx, cy = w/2, h/2
        camera_martix = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])
        
        return camera_martix
    
    def get_transformation_matrix(self):
        transformation_matrix = np.eye(3, 4)
        
        return transformation_matrix
    

    def get_bbox(self, keypoints_2d):
        left, top, right, down = (
            keypoints_2d[:, 0].min(),
            keypoints_2d[:, 1].min(),
            keypoints_2d[:, 0].max(),
            keypoints_2d[:, 1].max()
        )
                
        center_x, center_y = (left + right) / 2, (top + down) / 2
        w, h = right - left, down - top
        bbox = (
            center_x - w/2,
            center_y - h/2,
            center_x + w/2,
            center_y + h/2
        )

        if np.sum(bbox) == 0.0 or np.sum(np.isnan(bbox)) > 0:
            return np.array([0.0, 0.0, 100.0, 100.0])

        bbox = utils.common.get_square_bbox(bbox)
        bbox = utils.common.scale_bbox(bbox, self.bbox_scale)
        
        return bbox

    # def normalize_keypoints_2d(self, keypoints_2d, image_shape):
    #     convex_hull = scipy.spatial.ConvexHull(keypoints_2d[:, :2])
    #     center = np.mean(convex_hull.points[convex_hull.vertices], axis=0)

    #     keypoints_2d[:, :2] = keypoints_2d[:, :2] - center
    #     if self.keypoints_2d_normalization == 'area':
    #          keypoints_2d[:, :2] = keypoints_2d[:, :2] / np.sqrt(convex_hull.area)
    #     elif self.keypoints_2d_normalization == 'image_shape':
    #         keypoints_2d[:, :2] = keypoints_2d[:, :2] / np.array([image_shape[1], image_shape[0]])
    #     elif self.keypoints_2d_normalization == 'no':
    #         pass
    #     else:
    #         raise NotImplementedError("Unknown keypoints_2d_normalization mode: {self.keypoints_2d_normalization}")

    #     # norm depth
    #     if keypoints_2d.shape[1] == 3:  # 3d keypoints
    #         keypoints_2d[:, 2] /= 100.0

    #     return keypoints_2d

    def load_sample(self, identity_id, video_id, utterance_id, seq_index):
        sample = dict()
        sample['key'] = (identity_id, video_id, utterance_id, seq_index)
        
        # load h5_data
        try:
            h5_data = self.h5f[identity_id][video_id][utterance_id]
        except Exception as e:
            print(identity_id, video_id, utterance_id, seq_index)
            print(e)
        
        sample['expression'] = h5_data['expressions'][seq_index]
        sample['pose'] = h5_data['poses'][seq_index]  # 90 = [63 pose + 3 jaw + 6 eye + 12 hand + 3 trans + 3 root_orient]
        sample['beta'] = h5_data['betas'][:]

        sample['keypoints_2d_op'] = h5_data['face_keypoints_2d'][seq_index].astype(np.float32)
        
        # load image
        if self.return_images:
            image = self.load_image(identity_id, video_id, utterance_id, seq_index)
            orig_h, orig_w = image.shape[:2]

            # get keypoints_2d
            op_bbox = self.get_bbox(sample['keypoints_2d_op'])

            image_op_cropped = utils.common.crop_image(image, op_bbox)
            keypoints_3d, keypoints_3d_normed = self.face_mesh_model(image_op_cropped)
            
            if keypoints_3d_normed is None:
                keypoints_3d_normed = np.zeros((468, 3))
                keypoints_3d = np.zeros((468, 3))
                bbox = op_bbox
            else:
                keypoints_3d[:, :2] += np.array(op_bbox[:2])
                bbox = self.get_bbox(keypoints_3d[:, :2])

            if self.return_keypoints_3d:
                sample['keypoints'] = keypoints_3d_normed.astype(np.float32)
                sample['keypoints_orig'] = keypoints_3d.astype(np.float32)
            else:
                sample['keypoints'] = keypoints_3d_normed[:, :2].astype(np.float32)
                sample['keypoints_orig'] = keypoints_3d[:, :2].astype(np.float32)
            
            # crop
            image = utils.common.crop_image(image, bbox)
            
            # resize
            image = utils.common.resize_image(image, self.image_shape)
            
            image = image / 255.0
            image = image.transpose(2, 0, 1)
            
            sample['image'] = image
            
        # load projection matrix
        h, w = image.shape[1:3]
        bbox_h, bbox_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        
        if 'camera_matrix' in h5_data:
            camera_matrix = h5_data['camera_matrix'][:]
        else:
            camera_matrix = self.get_camera_matrix(orig_h, orig_w)

        camera_matrix = utils.common.update_after_crop_and_resize(
            camera_matrix, bbox, (w/bbox_w, h/bbox_h)
        )
        
        transformation_matrix = self.get_transformation_matrix()
        
        projection_matrix = camera_matrix @ transformation_matrix
        
        sample['camera_matrix'] = camera_matrix
        sample['projection_matrix'] = projection_matrix
        sample['h'] = h
        sample['w'] = w
        
        # update keypoints 2d after crop and resize
        sample['keypoints_orig'][:, 0] -= bbox[0]
        sample['keypoints_orig'][:, 1] -= bbox[1]
        
        sample['keypoints_orig'][:, 0] *= w/bbox_w
        sample['keypoints_orig'][:, 1] *= h/bbox_h

        # # normalize keypoints 2d
        # sample['keypoints_2d_orig'] = sample['keypoints_2d'].copy()
        # if not np.all(sample['keypoints_2d'] == 0.0):
        #     try:
        #         sample['keypoints_2d'] = self.normalize_keypoints_2d(sample['keypoints_2d'], (h, w)).astype(np.float32)
        #     except Exception as e:
        #         sample['keypoints_2d'] = np.zeros_like(sample['keypoints_2d']).astype(np.float32)
                
        return sample
    
    def __len__(self):
        return len(self.scheme)
    
    def __getitem__(self, index):
        # this should be normally done in __init__, but due to DataLoader behaviour 
        # when num_workers > 1, the h5 file is opened during first data access:
        # https://github.com/pytorch/pytorch/issues/11929#issuecomment-649760983

        if not hasattr(self, 'h5f'):
            self.open_h5_file()

        sample_key = self.scheme[index]
        sample = self.load_sample(*sample_key)
        return sample
    
    @staticmethod
    def build_scheme(h5f):
        scheme = []
        for identity_id in tqdm(h5f):
            for video_id in h5f[identity_id]:
                for utterance_id in h5f[identity_id][video_id]:
                    seq_length = h5f[identity_id][video_id][utterance_id]['expressions'].shape[0]
                    for seq_index in range(seq_length):
                        scheme.append((identity_id, video_id, utterance_id, seq_index))
                    
        scheme = sorted(scheme)     
        return scheme
    
    @staticmethod
    def preprocess_dataset(face_root, image_root, openpose_root, h5_path):
        # load scheme
        scheme = []

        identity_id_list = sorted(os.listdir(face_root))
        for identity_id in tqdm(identity_id_list):
            identity_dir = os.path.join(face_root, identity_id)
            video_id_list = sorted(os.listdir(identity_dir))

            for video_id in video_id_list:
                video_dir = os.path.join(identity_dir, video_id)
                utterance_id_list = sorted(os.listdir(video_dir))

                for utterance_id in utterance_id_list:
                    utterance_dir = os.path.join(video_dir, utterance_id)

                    scheme.append((identity_id, video_id, utterance_id))

        scheme = sorted(scheme)

        # build h5 file
        with h5py.File(h5_path, 'w') as hf:
            for (identity_id, video_id, utterance_id) in tqdm(scheme):
                # load face
                face_dir = os.path.join(face_root, identity_id, video_id, utterance_id, 'joints_op_face')

                expressions = np.load(os.path.join(face_dir, 'expressions.npy')) * 100
                poses = np.load(os.path.join(face_dir, 'poses.npy'))
                betas = np.load(os.path.join(face_dir, 'betas.npy'))

                # load openpose keypoints 2d
                openpose_dir = os.path.join(openpose_root, identity_id, video_id, utterance_id)

                face_keypoints_2d_list = []
                names = sorted(os.listdir(openpose_dir))
                for name in names:
                    path = os.path.join(openpose_dir, name)

                    with open(path) as f:
                        openpose_data = json.load(f)

                    face_keypoints_2d = openpose_data['people'][0]['face_keypoints_2d']
                    face_keypoints_2d = np.array(face_keypoints_2d).reshape(70, 3)
                    face_keypoints_2d = face_keypoints_2d[:, :2]  # remove confidences

                    face_keypoints_2d_list.append(face_keypoints_2d)

                face_keypoints_2d_arr = np.array(face_keypoints_2d_list)

                # save to h5
                group = hf.create_group(f"{identity_id}/{video_id}/{utterance_id}")
                group['expressions'] = expressions
                group['poses'] = poses
                group['betas'] = betas

                group['face_keypoints_2d'] = face_keypoints_2d_arr
                                

@hydra.main(config_path='config/default.yaml')
def main(config):
    print(config.pretty())

    # preprocess
    print(f"Preprocess split {split}")
    VoxCeleb2FaceDataset.preprocess_dataset(
        config.data.face_root, config.data.image_root, config.data.openpose_root, config.data.h5_path
    )

    # save scheme
    print("Build scheme")
    h5f = h5py.File(config.data.h5_path, mode='r', libver='latest')
    scheme = VoxCeleb2FaceDataset.build_scheme(h5f)
    with open(config.data.scheme_path, 'wb') as f:
        pickle.dump(scheme, f)

    # filter scheme
    print("Filter scheme")
    dataset = VoxCeleb2FaceDataset(
        config.data.h5_path, config.data.scheme_path,
        config.data.image_root,
        return_images=config.data.return_images, bbox_scale=config.data.bbox_scale, image_shape=config.data.image_shape
    )

    invalid_indices = []
    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]
        except Exception as e:
            invalid_indices.append(i)
            print(f"Index {i} is invalid. Reason: {e}")
            
    invalid_indices = set(invalid_indices)
    print(f"Found {len(invalid_indices)} invalid samples")

    scheme_filtered = [sample_key for i, sample_key in enumerate(dataset.scheme) if i not in invalid_indices]
    with open(config.data.scheme_path, 'wb') as f:
        pickle.dump(scheme_filtered, f)

    print("Success!")
    

if __name__ == '__main__':
    main()

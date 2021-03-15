import os
import argparse
import shutil
import json
import glob
from tqdm import tqdm
import multiprocessing

import numpy as np
import cv2

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib

from face_mesh_mediapipe import FaceMeshMediaPipe


anchors_path = "./models/face_anchors.csv"
detection_model_path = "./models/face_detection_front.tflite"
landmark_model_path = "./models/face_landmark.tflite"

FACE_MESH_MODEL = FaceMeshMediaPipe(anchors_path, detection_model_path, landmark_model_path)


KINECT_NOSE_INDEX = 27
KINECT_EYE_LEFT_INDEX = 28
KINECT_EAR_LEFT_INDEX = 29
KINECT_EYE_RIGHT_INDEX = 30
KINECT_EAR_RIGHT_INDEX = 31

KINECT_FACE_INDICES = [
    KINECT_NOSE_INDEX, 
    KINECT_EYE_LEFT_INDEX, 
    KINECT_EAR_LEFT_INDEX, 
    KINECT_EYE_RIGHT_INDEX,
    KINECT_EAR_RIGHT_INDEX
]


def load_P_rgb(camera_parameters):
    cam_rgb_int = np.eye(3)
    cam_rgb_int[0, 0] = camera_parameters["rgb_undistorted_intrinsics"]['fx']
    cam_rgb_int[1, 1] = camera_parameters["rgb_undistorted_intrinsics"]['fy']
    cam_rgb_int[0, 2] = camera_parameters["rgb_undistorted_intrinsics"]['cx']
    cam_rgb_int[1, 2] = camera_parameters["rgb_undistorted_intrinsics"]['cy']
    
    cam_d_int = np.eye(3)
    cam_d_int[0, 0] = camera_parameters["depth_undistorted_intrinsics"]['fx']
    cam_d_int[1, 1] = camera_parameters["depth_undistorted_intrinsics"]['fy']
    cam_d_int[0, 2] = camera_parameters["depth_undistorted_intrinsics"]['px']
    cam_d_int[1, 2] = camera_parameters["depth_undistorted_intrinsics"]['py']
    
    posevec = np.zeros(6)
    for i in range(0, 3):
        posevec[i] = camera_parameters["depth_to_rgb"]["r"][i]
    for i in range(0, 3):
        posevec[i+3] = camera_parameters["depth_to_rgb"]["t"][i]
        
    rot_mat, jac = cv2.Rodrigues(posevec[0:3])
    depth_to_rgb = np.eye(4)
    depth_to_rgb[0:3, 0:3] = rot_mat
    depth_to_rgb[0:3, 3] = posevec[3:6] / 1000

    T_id = np.eye(4)

    P_rgb = cam_rgb_int @ depth_to_rgb[0:3, 0:4]
    return P_rgb


def crop_bbox_by_keypoints_2d(keypoints_2d):
    x_min, y_min = np.min(keypoints_2d, axis=0)
    x_max, y_max = np.max(keypoints_2d, axis=0)

    width = x_max - x_min
    height = y_max - y_min

    # size = max(width, height)
    bbox = x_min, y_min, x_min + width, y_min + height
    bbox = get_square_bbox(bbox)

    bbox = scale_bbox(bbox, 3)
    
    return bbox 


def crop_image(image, bbox):
    """Crops area from image specified as bbox. Always returns area of size as bbox filling missing parts with zeros
    Args:
        image numpy array of shape (height, width, 3): input image
        bbox tuple of size 4: input bbox (left, upper, right, lower)
    Returns:
        cropped_image numpy array of shape (height, width, 3): resulting cropped image
    """

    image_pil = Image.fromarray(image)
    image_pil = image_pil.crop(bbox)

    return np.asarray(image_pil)


def get_square_bbox(bbox):
    """Makes square bbox from any bbox by stretching of minimal length side
    Args:
        bbox tuple of size 4: input bbox (left, upper, right, lower)
    Returns:
        bbox: tuple of size 4:  resulting square bbox (left, upper, right, lower)
    """

    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    if width > height:
        y_center = (upper + lower) // 2
        upper = y_center - width // 2
        lower = upper + width
    else:
        x_center = (left + right) // 2
        left = x_center - height // 2
        right = left + height

    return left, upper, right, lower


def scale_bbox(bbox, scale):
    left, upper, right, lower = bbox
    width, height = right - left, lower - upper

    x_center, y_center = (right + left) // 2, (lower + upper) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_upper = y_center - new_height // 2
    new_lower = new_upper + new_height

    return new_left, new_upper, new_right, new_lower


def process_seq(
    frame_root, bt_root, camera_parameters_root, result_root,
    subject_id, camera_id,
    save_images=False):

    frame_dir = os.path.join(frame_root, subject_id, camera_id, "color_undistorted")
    bt_path = os.path.join(bt_root, subject_id, "bt.json")
    camera_parameters_path = os.path.join(camera_parameters_root, subject_id, camera_id, "mkv_meta.json")
    result_dir = os.path.join(result_root, subject_id, camera_id)

    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir, exist_ok=True)

    with open(bt_path) as f:
        keypoints_data = json.load(f)

    with open(camera_parameters_path) as f:
        camera_parameters = json.load(f)
        
    
    P_rgb = load_P_rgb(camera_parameters)
    
    result = dict()
    for i in tqdm(range(len(keypoints_data))):
        frame_index = keypoints_data[i]['frame_index']
        frame_path = os.path.join(frame_dir, f"{frame_index:06}.jpg")

        try:
            image = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Failed to load {frame_path}. Reason: {e}")
            continue

        # get wide face crop
        try:
            kinect_bodies = keypoints_data[i]['uids'][camera_id]['bodies']
            kinect_body = kinect_bodies[0]
        except Exception as e:
            print(f"Failed to load kinect keypoints for {subject_id}:{camera_id}. Reason: {e}")
            continue
            
        keypoints_3d = np.array(kinect_body['joint_positions'])
        
        keypoints_3d_homo = np.pad(keypoints_3d, (0, 1), constant_values=1.0)
        keypoints_2d = keypoints_3d_homo @ P_rgb.T
        keypoints_2d = keypoints_2d[:, 0:2] / keypoints_2d[:, 2].reshape(-1, 1)

        face_bbox = crop_bbox_by_keypoints_2d(keypoints_2d[KINECT_FACE_INDICES])
        image_croppped = crop_image(image, face_bbox)

        keypoints_2d = FACE_MESH_MODEL(image_croppped)
        if keypoints_2d is None:
            keypoints_2d = np.zeros((468, 2))

        keypoints_2d += np.array([face_bbox[0], face_bbox[1]])
        
        result[f"{frame_index:06d}"] = keypoints_2d.tolist()
        
        if save_images:
            canvas = image.copy()
            for point in keypoints_2d:
                x, y = int(point[0]), int(point[1])
                canvas = cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)

            # save
            cv2.imwrite(os.path.join(result_dir, f"{frame_index}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    
    result_path = os.path.join(result_dir, "face_mesh_mediapipe.json")
    with open(result_path, 'w') as f:
        json.dump(result, f)

    print(f"{subject_id}:{camera_id} Success")

def process_seq_star(task):
    return process_seq(*task)


def main(args):
    frame_root = "/Vol1/dbstore/datasets/violet/AzurePeople/offline_processor2/align_06"
    bt_root = "/Vol1/dbstore/datasets/violet/AzurePeople/offline_processor2/align_02"
    camera_parameters_root = "/Vol1/dbstore/datasets/violet/AzurePeople/offline_processor2/align_02"
    result_root = "/Vol1/dbstore/datasets/k.iskakov/azure_people_processed/face_mesh_mediapipe_04_aug"

    # setup multi-cpu
    n_jobs = args.n_jobs
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count() - 1

    print("Using {} CPUs".format(n_jobs))

    # generate tasks
    tasks = []

    subject_ids = sorted(os.listdir(frame_root))
    for subject_id in subject_ids:
        camera_ids = sorted(os.listdir(os.path.join(frame_root, subject_id)))
        camera_ids = filter(lambda camera_id: os.path.isdir(os.path.join(frame_root, subject_id, camera_id)), camera_ids)

        bt_path = os.path.join(bt_root, subject_id, "bt.json")
        for camera_id in camera_ids:
            frame_dir = os.path.join(frame_root, subject_id, camera_id, "color_undistorted")
            camera_parameters_path = os.path.join(camera_parameters_root, subject_id, camera_id, "mkv_meta.json") 
            result_dir = os.path.join(result_root, subject_id, camera_id)
                
            tasks.append((
                frame_root, bt_root, camera_parameters_root, result_root,
                subject_id, camera_id,
                False
            ))

    with multiprocessing.Pool(n_jobs) as pool:
        for _ in tqdm(pool.imap(process_seq_star, tasks), total=len(tasks)):
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', type=int, default=-1, help="Number of jobs to use (pass '-1' to use all available - 1)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    main(args)

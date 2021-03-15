import os
import random
import time

import numpy as np
import cv2
from PIL import Image

import torch


def get_data_sampler(dataset, shuffle, is_distributed):
    if is_distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


def dict2device(d, device, dtype=None):
    if isinstance(d, np.ndarray):
        d = torch.from_numpy(d)

    if torch.is_tensor(d):
        d = d.to(device)
        if dtype is not None:
            d = d.type(dtype)
        return d

    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = dict2device(v, device, dtype=dtype)

    return d


def get_square_bbox(bbox):
    left, top, right, down = bbox
    width, height = right - left, top - down

    if width > height:
        y_center = (top + down) // 2
        top = y_center - width // 2
        down = top + width
    else:
        x_center = (left + right) // 2
        left = x_center - height // 2
        right = left + height

    return left, top, right, down


def scale_bbox(bbox, scale):
    left, top, right, down = bbox
    width, height = right - left, down - top

    x_center, y_center = (right + left) // 2, (down + top) // 2
    new_width, new_height = int(scale * width), int(scale * height)

    new_left = x_center - new_width // 2
    new_right = new_left + new_width

    new_top = y_center - new_height // 2
    new_down = new_top + new_height

    return new_left, new_top, new_right, new_down


# def crop_image(image, bbox):
#     image_pil = Image.fromarray(image)
#     image_pil = image_pil.crop(bbox)

#     return np.asarray(image_pil)

def crop_image(image, bbox):
    h, w = image.shape[:2]
    left, top, right, bottom = (int(x) for x in bbox)
    
    assert (left < right) and (top < bottom)
    assert (left < w) and (top < h) and (right > 0) and (bottom > 0)
    
    # crop
    image = image[
        max(top, 0):min(bottom, h),
        max(left, 0):min(right, w),
        :
    ]
    
    # pad image    
    left_pad = max(0 - left, 0)
    top_pad = max(0 - top, 0)
    right_pad = max(right - w, 0)
    bottom_pad = max(bottom - h, 0)
    
    if any((left_pad, top_pad, right_pad, bottom_pad)):
        image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)))

    
    return image


def resize_image(image, shape):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)

def scale_image(image, scale_factor):
    shape = image.shape[:2]
    new_shape = (int(scale_factor * shape[0]), int(scale_factor * shape[1]))

    image = resize_image(image, new_shape)

    return image


def update_after_crop_and_resize(camera_matrix, bbox, scale):
    left, top, right, down = bbox
    scale_x, scale_y = scale

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # build new camera matrix
    new_fx = scale_x * fx
    new_fy = scale_y * fy
    new_cx = scale_x * (cx - left)
    new_cy = scale_y * (cy - top)

    new_camera_matrix = np.array([
        [new_fx, 0.0, new_cx],
        [0.0, new_fy, new_cy],
        [0.0, 0.0, 1.0]
    ])

    return new_camera_matrix


def setup_environment(seed):
    # random
    random.seed(seed)

    # numpy
    np.random.seed(seed)
    
    # cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # pytorch
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


def squeeze_metrics(d):
    metrics = dict()
    for k, v in d.items():
        if torch.is_tensor(v):
            metrics[k] = v.mean().item()
        elif isinstance(v, float):
            metrics[k] = v
        else:
            raise NotImplementedError("Unknown datatype for metric: {}".format(type(v)))

    return metrics


def reduce_metrics(metrics):
    metrics_dict = dict()
    for k in metrics[0].keys():
        metrics_dict[k] = np.mean([item[k] for item in metrics])

    return metrics_dict


def get_lastest_checkpoint(checkpoint_dir):
    checkpoint_name = sorted(os.listdir(checkpoint_dir))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    return checkpoint_path


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def blend_2_images_with_alpha(background, overlay, alpha=0.5):
    if overlay.shape[-1] == 4:  # RGBA
        overlay, mask = overlay[:, :, :3], overlay[:, :, 3:]
        mask = mask > 0
    else:
        mask = np.ones((*background.shape[:2], 1))
        
    canvas = np.zeros_like(background)
    canvas =  (alpha * background + (1 - alpha) * mask * overlay).astype(np.uint8)
    
    return canvas


class Timer(object):
    def __init__(self, title='', verbose=True):
        self.title = title
        self.verbose = verbose
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        end = time.time()
        self.elapsed_time_seconds = end - self.start
        if self.verbose:
            print(f"{self.title}: {self.elapsed_time_seconds:.3f}/{1/self.elapsed_time_seconds:.1f} (time/fps)")
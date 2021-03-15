import pickle
import time
import numpy as np
import cv2
import json
import os
import os.path as osp

from multiprocessing_pipeline import Assembler, Processor, Dissembler
from multiprocessing_pipeline import QueueMsg, QueueData, MetaMsg

import pyk4a
from pyk4a import Config, PyK4A, ColorResolution

from face_expression.cropper import Cropper


def cam_params_decode(cam_params):
    K_rgb_distorted = np.eye(3, dtype=np.float32)
    for [i, j, k] in [[0, 0, 'fx'], [1, 1, 'fy'], [0, 2, 'cx'], [1, 2, 'cy']]:
        K_rgb_distorted[i, j] = cam_params['rgb_intrinsics'][k]

    rgb_distortion = np.array([
        cam_params['rgb_intrinsics'][k]
        for k in ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6']
    ], dtype=np.float32)

    rvec = np.array([
        cam_params['depth_to_rgb']['r'][i]
        for i in range(3)
    ], dtype=np.float32)

    tvec = np.array([
        cam_params['depth_to_rgb']['t'][i]
        for i in range(3)
    ], dtype=np.float32) / 1000

    rotmtx = cv2.Rodrigues(np.array(rvec))[0]

    return {
        'K_rgb_distorted': K_rgb_distorted,
        'rgb_distortion': rgb_distortion,
        'rvec': rvec,
        'rotmtx': rotmtx,
        'tvec': tvec,
        'color_resolution': cam_params['color_resolution']
    }


def get_cube(p, s=0.15):
    diffs = np.array([
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, 1, -1, 1, -1, 1, -1, 1]
    ], dtype=np.float32) * s

    return (diffs + p.reshape(3, 1)).T


class K4AProcessor(Processor):
    def __init__(
            self,
            name, msg_queue, input_queue, output_queue, assembler_input_queue,
            data_dirpath, processor_config,
            **kwargs
    ):
        Processor.__init__(self, name, msg_queue, input_queue, output_queue, assembler_input_queue, **kwargs)

        self.data_dirpath = data_dirpath

        self.fps = processor_config['fps']
        self.kinect_dump_fp = processor_config.get('dump_fp', None)
        self.realtime = self.kinect_dump_fp is None
        self.skip_old_atol_ms = 0.2 \
            if processor_config.get('skip_old', False) else None
        self.get_color_timestamp = processor_config.get(
            'get_color_timestamp', True)
        self.parallel_bt = processor_config.get('parallel_bt', True)
        self.log_level = processor_config.get('log_level', 2)
        self.gpu_id = processor_config.get('gpu_id', 0)

        self.face_cropper = None

        self.prev_time = time.time()
        self.k4a = None
        self.kinect_dump_frames = None
        self.decoded_cam_params = None
        self.index = -1
        self.dump_index = -1
        self.output_count = 0

    def process_value(self):
        self.index += 1
        init_done = False
        if self.realtime:
            if self.k4a is None:
                if self.fps == 30:
                    fps_config = pyk4a.FPS.FPS_30
                elif self.fps == 15:
                    fps_config = pyk4a.FPS.FPS_15
                else:
                    raise Exception(f'fps {self.fps} not found')
                k4a = PyK4A(Config(
                    color_resolution=ColorResolution.RES_1536P,
                    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                    camera_fps=fps_config,
                    gpu_id=self.gpu_id
                ))
                k4a.connect(lut=True)
                self.decoded_cam_params = cam_params_decode(
                    k4a.get_cam_params())
                self.k4a = k4a
                init_done = True
        elif self.kinect_dump_frames is None:
            with open(self.kinect_dump_fp, 'rb') as f:
                kinect_dump_data = pickle.load(f)
            self.decoded_cam_params = cam_params_decode(
                kinect_dump_data['cam_params'])
            self.kinect_dump_frames = kinect_dump_data['frames']
            init_done = True

        if init_done:
            self.msg_queue.put(MetaMsg(
                sender_subblock_name=self.subblock_name,
                acceptor_name='aggregate',
                acceptor_type='processor',
                msg={
                    'decoded_cam_params': self.decoded_cam_params
                }
            ))

        if self.realtime:
            d = self.k4a.get_capture2(
                verbose=self.fps if self.log_level >= 2 else 0,
                skip_old_atol_ms=self.skip_old_atol_ms,
                get_color_timestamp=self.get_color_timestamp,
                get_color=True, undistort_color=False,
                get_depth=False,
                get_bt=True, undistort_bt=False,
                parallel_bt=self.parallel_bt
            )
        else:
            self.dump_index += 1
            if self.dump_index >= len(self.kinect_dump_frames):
                self.dump_index = 0
                if self.log_level >= 1:
                    print('dump finished')
            d = self.kinect_dump_frames[self.dump_index]
            cur_time = time.time()
            sleep_time = max(0, 1 / self.fps - (cur_time - self.prev_time))
            time.sleep(sleep_time)
            self.prev_time = time.time()

        for p in ['pose', 'color']:
            if not p in d:
                return None

        if self.kinect_dump_fp is None:
            self.index += d['skip_count']
        #         print(d['depth_timestamp'], d['color_timestamp'])

        # cv2.imshow('a', d['color_undistorted'][::4, ::4, :])
        # cv2.waitKey(1)

        if self.face_cropper is None:
            self.face_cropper = Cropper(
                self.decoded_cam_params['rvec'],
                self.decoded_cam_params['tvec'],
                self.decoded_cam_params['K_rgb_distorted'],
                self.decoded_cam_params['rgb_distortion'])

        start = time.time()
        img = d['color'][..., :3]

        if len(d['pose']) < 1:
            return None

        body_pose = d['pose'][0, :, 2:5] / 1000
        body_conf = d['pose'][0, :, -1]

        hand_dist = np.sqrt(np.sum((body_pose[8] - body_pose[15]) ** 2))
        is_close_hands = hand_dist < 0.2

        w = img.shape[1]
        h = img.shape[0]
        hand_crops = dict()
        for hand_index, side in [
            [15, 'right'],
            [8, 'left'],
        ]:
            p = body_pose[hand_index]
            p_cube = get_cube(p)
            p_proj, _ = cv2.projectPoints(
                p_cube,
                self.decoded_cam_params['rvec'],
                self.decoded_cam_params['tvec'],
                self.decoded_cam_params['K_rgb_distorted'],
                self.decoded_cam_params['rgb_distortion'])
            p_proj = p_proj.reshape(-1, 2)
            lu = np.min(p_proj, axis=0)
            rb = np.max(p_proj, axis=0)
            lu = np.clip(lu, a_min=(0, 0), a_max=[w, h]).astype(int)
            rb = np.clip(rb, a_min=(0, 0), a_max=[w, h]).astype(int)
            crop = img[lu[1]:rb[1], lu[0]:rb[0], :3]
            crop = cv2.resize(crop, (128, 128))
            hand_crops[side] = crop

        face_crop, face_bbox = self.face_cropper.forward(img, body_pose)
        # cv2.imshow('face', face_crop)
        # cv2.waitKey(1)

        body_pose = body_pose @ self.decoded_cam_params['rotmtx'].T + \
                    self.decoded_cam_params['tvec']

        res = dict({
            'face_crop': face_crop,
            'hand_crops': hand_crops,
            'body_pose': body_pose,
            'body_conf': body_conf,
            'is_close_hands': is_close_hands,
            'time': start
        })

        if self.output_count == 0:
            if self.log_level >= 1:
                print(f'{self.subblock_name} working')
        self.output_count += 1

        return self.index, res

    def destructor(self):
        if self.k4a is not None:
            self.k4a.disconnect()


class K4ADissembler(Dissembler):
    def __init__(
            self,
            name, msg_queue, input_queue, output_queues,
            deepcopy=False,
            **kwargs
    ):
        Dissembler.__init__(self, name, msg_queue, input_queue, output_queues, **kwargs)
        self.deepcopy = deepcopy

    def process_queue_el(self, x):
        result = []
        for output_name in self.outputs:
            if output_name.startswith('face'):
                queue_data = QueueData(name=x.name, index=x.index, value=x.value['face_crop'])
                result.append(queue_data)
            elif output_name.startswith('hand'):
                queue_data = QueueData(name=x.name, index=x.index, value=x.value['hand_crops'])
                result.append(queue_data)
            elif output_name.startswith('body'):
                queue_data = QueueData(name=x.name, index=x.index, value={
                    'body_pose': x.value['body_pose'],
                    'body_conf': x.value['body_conf'],
                    'is_close_hands': x.value['is_close_hands'],
                    'time': x.value['time']
                })
                result.append(queue_data)
            else:
                raise Exception()
        return result

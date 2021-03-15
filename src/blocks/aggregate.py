from copy import deepcopy
import socket
import pickle
import threading
import queue as Queue
import os
import os.path as osp
import inspect
import sys
import torch
import cv2
import shutil
import time
import numpy as np

from multiprocessing_pipeline import Assembler, Processor, Dissembler
from multiprocessing_pipeline import QueueEl, QueueMsg, QueueData, MetaMsg

from smplx_kinect.common.smplx_vis import VisualizerMeshSMPLX
from smplx_kinect.common.body_models import HandsSMPLXWrapper


from body_pose.filterer import Filterer


class AggregateAssembler(Assembler):
    def __init__(self, name, msg_queue, input_queue, output_queue, input_names):
        Assembler.__init__(self, name, msg_queue, input_queue, output_queue)
        self.input_names = input_names
        self.inputs = dict()
        self.max_index = 0
        self.hungry_count = 1

    def custom_run(self):
        while True:
            result_queue_el = None
            while True:
                self.log('queue_wait', None)
                input_queue_el = self.input_queue.get()
                self.log('input_queue.get', input_queue_el)
                if input_queue_el is None:
                    break
                assert isinstance(input_queue_el, QueueEl), self.subblock_name
                if isinstance(input_queue_el, QueueMsg) and \
                        isinstance(input_queue_el.msg, str) and \
                        input_queue_el.msg == 'processor_fed':
                    self.hungry_count += 1
                    continue
                elif isinstance(input_queue_el, QueueMsg):
                    continue
                elif isinstance(input_queue_el, QueueData):
                    if input_queue_el.name in self.input_names:
                        index = input_queue_el.index
                        if index not in self.inputs:
                            self.inputs[index] = dict()
                        self.inputs[index][input_queue_el.name] = input_queue_el.value
                        indexes = list(sorted(filter(
                            lambda k: len(self.inputs[k]) == len(self.input_names),
                            self.inputs.keys()
                        )))
                        if len(indexes) > 0:
                            max_index = max(indexes)
                            self.max_index = max(self.max_index, max_index)
                            max_index_result_value = deepcopy(self.inputs[max_index])
                            delete_indexes = list(filter(
                                lambda x: x <= self.max_index,
                                self.inputs.keys()))
                            for delete_index in delete_indexes:
                                del self.inputs[delete_index]
                            if max_index == self.max_index:
                                result_queue_el = QueueData(
                                    name=self.name,
                                    index=max_index,
                                    value=max_index_result_value)
                                if self.hungry_count > 0:
                                    break

                    else:
                        # print(f'output {input_queue_el.name}')
                        # self.output_queue.put(input_queue_el)
                        print(f'Input source {input_queue_el.name} for block {self.subblock_name} not recognized')
                else:
                    raise Exception(
                        f'contact developer, no code for {type(input_queue_el)} in subblock {self.subblock_name}')
                self.log('tmp', None)

            assert self.hungry_count > 0
            self.hungry_count = 0
            self.log('output_queue.put', result_queue_el)
            self.output_queue.put(result_queue_el)
            self.log('tmp', None)


class AggregateProcessor(Processor):
    def __init__(
            self,
            name, msg_queue, input_queue, output_queue, assembler_input_queue,
            data_dirpath, processor_config, pykinect_data_dp, output_dirpath,
            **kwargs
    ):
        Processor.__init__(self, name, msg_queue, input_queue, output_queue, assembler_input_queue, **kwargs)
        self.data_dirpath = data_dirpath
        self.processor_config = processor_config
        self.log_level = processor_config.get('log_level', 2)
        self.pykinect_data_dp = pykinect_data_dp
        self.filterer_config = self.processor_config['filterer']
        self.filterer_config['body_models_dp'] = osp.join(
            self.pykinect_data_dp, 'body_models')
        self.output_dirpath = output_dirpath

        self.gender = self.processor_config['gender']
        self.beta = np.load(osp.join(
            self.data_dirpath,
            self.processor_config['person_shape_path'],
            'betas.npy'
        ))

        self.vis_config = self.processor_config['vis_pose']
        self.smplx_wrapper = None
        self.visualizer = None
        if self.vis_config['enable'] and self.vis_config['imsave']:
            self.vis_dp = osp.join(self.output_dirpath, 'aggregate')
            os.makedirs(self.vis_dp, exist_ok=True)
        else:
            self.vis_dp = None

        self.filterer = None
        self.decoded_cam_params = None
        self.smplx_device = None
        self.visualize_device = None
        self.prev_time = time.time()

        self.index = -1
        self.output_count = 0

    def process_value(self, x):
        self.index += 1

        if isinstance(x, QueueMsg):
            k = 'decoded_cam_params'
            if k in x.msg:
                self.decoded_cam_params = x.msg[k]
            return
        else:
            if self.decoded_cam_params is None:
                return

        if self.filterer is None:
            self.smplx_device = torch.device(self.filterer_config['device'])
            self.filterer = Filterer(**self.filterer_config)
            if self.vis_config['enable']:
                self.visualize_device = torch.device(self.vis_config['device'])

                self.smplx_wrapper = HandsSMPLXWrapper(
                    body_models_dp=self.filterer_config['body_models_dp'],
                    use_pca=False, device=self.smplx_device, flat_hand_mean=True
                )
                self.visualizer = VisualizerMeshSMPLX(
                    K=self.decoded_cam_params['K_rgb_distorted'],
                    scale=self.vis_config.get('scale', 1),
                    size=self.decoded_cam_params['color_resolution'],
                    device=self.visualize_device,
                    smplx_wrapper=self.smplx_wrapper
                )

        x = self.filterer.filter(x)

        if self.vis_config['enable']:
            smplx_output = self.smplx_wrapper.get_output(
                gender=self.gender,
                betas=self.beta,
                body_pose=x['body_pose'],
                rvec=x['global_rot'],
                tvec=x['global_trans'],
                left_hand_pose=x['left_hand_pose'],
                right_hand_pose=x['right_hand_pose'],
                expression=x.get('face_expression', None),
                jaw_pose=x.get('jaw_pose', None)
            )

            vertices = torch.tensor(smplx_output.vertices[0].contiguous()).to(
                device=self.visualize_device)
            # vertices = vertices @ cv2.Rodrigues(np.array([np.pi, 0, 0]))[0].T
            vis = self.visualizer.vertices2vis(vertices)

            if self.vis_config['imsave']:
                cv2.imwrite(
                    osp.join(self.vis_dp, f'{self.index:06d}.jpg'), vis)

            if self.vis_config['imshow']:
                cv2.imshow('a', vis)
                cv2.waitKey(1)

        cur = time.time()

        if self.log_level >= 2:
            print(f'aggregate output2output timing: {(cur - self.prev_time) * 1000:.01f}ms')
        self.prev_time = cur

        if self.output_count == 0:
            if self.log_level >= 1:
                print(f'{self.subblock_name} working')
        self.output_count += 1

    def destructor(self):
        pass
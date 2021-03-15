import os
import os.path as osp
import sys
import inspect
import cv2

from multiprocessing_pipeline import Assembler, Processor, Dissembler
from multiprocessing_pipeline import QueueMsg, QueueData, MetaMsg

from minimal_hand.wrappers import ModelPipeline
from minimal_hand.kinematics import mpii_to_mano

import numpy as np


class HandPose(Processor):
    def __init__(
        self,
        name, msg_queue, input_queue, output_queue, assembler_input_queue,
        data_dirpath, processor_config,
        **kwargs
    ):
        Processor.__init__(
            self,
            name, msg_queue, input_queue, output_queue, assembler_input_queue,
            **kwargs)
        self.data_dirpath = data_dirpath
        self.processor_config = processor_config
        self.log_level = processor_config.get('log_level', 2)

        self.model = None
        self.flip_side = 'right'

        self.index = -1
        self.output_count = 0

    def process_value(self, x):
        self.index += 1
        if self.model is None:
            self.model = ModelPipeline(
                HAND_MESH_MODEL_PATH=osp.join(
                    self.data_dirpath,
                    self.processor_config['hand_mesh_model_path']),
                DETECTION_MODEL_PATH=osp.join(
                    self.data_dirpath,
                    self.processor_config['detection_model_path']),
                IK_MODEL_PATH=osp.join(
                    self.data_dirpath,
                    self.processor_config['ik_model_path']),
                gpu_id=self.processor_config['gpu_id']
            )

        result = dict()
        for side, crop in x.items():
            # cv2.imshow(side, crop)
            # cv2.waitKey(1 if side == 'left' else 2)

            if side == self.flip_side:
                crop = np.flip(crop, axis=1)

            _, theta_mpii = self.model.process(crop)
            theta_mano = mpii_to_mano(theta_mpii)
            result[side] = theta_mano

        if self.output_count == 0:
            if self.log_level >= 1:
                print(f'{self.subblock_name} working')
        self.output_count += 1

        return result

    def destructor(self):
        pass

import os.path as osp
import numpy as np
import time
import torch

from multiprocessing_pipeline import Assembler, Processor, Dissembler
from multiprocessing_pipeline import QueueMsg, QueueData, MetaMsg

from body_pose.inferer import Inferer


class BodyPose(Processor):
    def __init__(
        self,
        name, msg_queue, input_queue, output_queue, assembler_input_queue,
        data_dirpath, processor_config, pykinect_data_dp,
        **kwargs
    ):
        Processor.__init__(self, name, msg_queue, input_queue, output_queue, assembler_input_queue, **kwargs)
        self.data_dirpath = data_dirpath
        self.processor_config = processor_config
        self.pykinect_data_dp = pykinect_data_dp
        self.log_level = processor_config.get('log_level', 2)
        self.device = torch.device(self.processor_config['device'])

        self.gender = self.processor_config['gender']
        self.beta = np.load(osp.join(
            self.data_dirpath,
            self.processor_config['person_shape_path'],
            'betas.npy'
        ))
        self.model_dp = osp.join(self.data_dirpath,
                            self.processor_config['model_path'])
        self.inferer = None

        self.index = -1
        self.output_count = 0

    def process_value(self, x):
        self.index += 1
        if self.inferer is None:
            self.inferer = Inferer(
                model_dp=self.model_dp,
                checkpoint=self.processor_config['checkpoint'],
                device=self.device,
                beta=self.beta,
                gender=self.gender,
                pykinect_data_dp=self.pykinect_data_dp
            )

        kinect_joints = x['body_pose']
        kinect_confs = x['body_conf']

        inf_result = self.inferer.inf(kinect_joints, kinect_confs)

        inf_result['time'] = x['time']
        inf_result['is_close_hands'] = x['is_close_hands']

        if self.output_count == 0:
            if self.log_level >= 1:
                print(f'{self.subblock_name} working')
        self.output_count += 1

        return inf_result

    def destructor(self):
        pass

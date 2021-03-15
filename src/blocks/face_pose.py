import os.path as osp

from multiprocessing_pipeline import Assembler, Processor, Dissembler
from multiprocessing_pipeline import QueueMsg, QueueData, MetaMsg

from face_expression.inferer import Inferer


class FacePose(Processor):
    def __init__(
        self,
        name, msg_queue, input_queue, output_queue, assembler_input_queue,
        data_dirpath, processor_config,
        **kwargs
    ):
        Processor.__init__(self, name, msg_queue, input_queue, output_queue, assembler_input_queue, **kwargs)
        self.data_dirpath = data_dirpath
        self.processor_config = processor_config
        self.log_level = processor_config.get('log_level', 2)
        self.device = self.processor_config['device']

        self.inferer = None

        self.index = -1
        self.output_count = 0

    def process_value(self, x):
        self.index += 1
        if self.inferer is None:
            config_path = osp.join(
                self.data_dirpath, self.processor_config['config_path'])
            checkpoint_path = osp.join(
                self.data_dirpath, self.processor_config['checkpoint_path'])

            self.inferer = Inferer(
                config_path,
                checkpoint_path,
                device=self.device
            )

        try:
            pred = self.inferer.forward(x, beta=None)
            expression_pred, jaw_pose_pred, keypoints_2d = pred
        except:
            return None

        result = {
            'expression': expression_pred,
            'jaw_pose': jaw_pose_pred
        }

        if self.output_count == 0:
            if self.log_level >= 1:
                print(f'{self.subblock_name} working')
        self.output_count += 1

        return result

    def destructor(self):
        pass

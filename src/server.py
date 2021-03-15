import os
import os.path as osp
import datetime
import argparse
import yaml

# Important! Set import order
import pyk4a
import minimal_pytorch_rasterizer
import torch

from blocks.k4a import K4AProcessor, K4ADissembler
from blocks.body_pose import BodyPose
from blocks.aggregate import AggregateAssembler, AggregateProcessor

from multiprocessing_pipeline import Block, Pipeline
from multiprocessing_pipeline import DummySkipAssembler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    return args


def start_server(config, output_dirpath):
    pykinect_data_dp = osp.join(config['data_dirpath'], 'pykinect')

    is_hand_pose = config['hand_pose']['enable']
    is_face_pose = config['face_pose']['enable']
    enabled = ['body_pose']
    if is_hand_pose:
        from blocks.hand_pose import HandPose
        enabled.append('hand_pose')
    if is_face_pose:
        from blocks.face_pose import FacePose
        enabled.append('face_pose')

    p = Pipeline(check_cycles=True)
    p.add_block(Block('k4a', use_assembler=False, use_dissembler=True))
    p.add_block(Block('body_pose', use_dissembler=False))
    if is_hand_pose:
        p.add_block(Block('hand_pose', use_dissembler=False))
    if is_face_pose:
        p.add_block(Block('face_pose', use_dissembler=False))
    p.add_block(Block('aggregate', use_dissembler=False))

    p.set_outputs('k4a', enabled)
    p.set_outputs('body_pose', ['aggregate'])
    if is_hand_pose:
        p.set_outputs('hand_pose', ['aggregate'])
    if is_face_pose:
        p.set_outputs('face_pose', ['aggregate'])

    p.check_connections()
    p.create_queues()

    p.set_processor(
        'k4a', K4AProcessor,
        data_dirpath=config['data_dirpath'],
        processor_config=config['k4a']
    )
    p.set_dissembler('k4a', K4ADissembler, deepcopy=False)

    p.set_assembler('body_pose', DummySkipAssembler)
    p.set_processor(
        'body_pose', BodyPose,
        data_dirpath=config['data_dirpath'],
        processor_config=config['body_pose'],
        pykinect_data_dp=pykinect_data_dp
    )

    if is_hand_pose:
        p.set_assembler('hand_pose', DummySkipAssembler)
        p.set_processor(
            'hand_pose', HandPose,
            data_dirpath=config['data_dirpath'],
            processor_config=config['hand_pose']
        )

    if is_face_pose:
        p.set_assembler('face_pose', DummySkipAssembler)
        p.set_processor(
            'face_pose', FacePose,
            data_dirpath=config['data_dirpath'],
            processor_config=config['face_pose']
        )

    p.set_assembler(
        'aggregate', AggregateAssembler,
        input_names=enabled
    )
    p.set_processor(
        'aggregate', AggregateProcessor,
        data_dirpath=config['data_dirpath'],
        processor_config=config['aggregate'],
        pykinect_data_dp=pykinect_data_dp,
        output_dirpath=output_dirpath
    )

    p.start(log_dirpath=osp.join(output_dirpath, 'log'))


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dirpath = osp.join(
        config['output_dirpath'],
        datetime.datetime.strftime(
            datetime.datetime.now(),
            "%Y.%m.%d_%H:%M:%S"))
    os.makedirs(output_dirpath, exist_ok=True)

    start_server(config, output_dirpath)


if __name__ == "__main__":
    main()

import os
from abc import ABC, abstractmethod
import datetime

import numpy as np
import yaml
from omegaconf import OmegaConf

import wandb

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from face_expression import utils


# class AbstractLogger(ABC):
#     @abstractmethod
#     def __init__(self, config, logdir, project_name, experiment_name):
#         pass

#     @abstractmethod
#     def add_image(self, label, image, step, prefix=''):
#         pass

#     @abstractmethod
#     def add_scalar(self, label, value, step, prefix=''):
#         pass

#     @abstractmethod
#     def add_scalar_dict(self, scalar_dict, step, prefix=''):
#         pass

#     @property
#     def experiment_dir(self):
#         pass

#     @property
#     def experiment_name(self):
#         pass



# class TesorboardLogger(AbstractLogger):
#     def __init__(self, config, logdir, project_name, experiment_name):
#         # setup experiment dir
#         self._experiment_name = datetime.datetime.now().strftime(f"{experiment_name}@%m-%d_%H-%M")
#         self._experiment_dir = os.path.join(logdir, project_name, self.experiment_name)
#         os.makedirs(self.experiment_dir, exist_ok=True)

#         # setup writer
#         self.writer = SummaryWriter(self.experiment_dir)

#         # add config
#         self._add_config(config)

#     def _add_config(self, config):
#         config_path = os.path.join(self.experiment_dir, "config.yaml")
#         with open(config_path, 'w') as f:
#             config_dict = omegaconf.OmegaConf.to_container(config)
#             yaml.safe_dump(config_dict, f, default_flow_style=False)

#     def add_image(self, label, image, step, prefix=''):
#         label = f"{prefix}/{label}"
#         self.writer.add_image(label, image, step)

#     def add_scalar(self, label, value, step, prefix=''):
#         if hasattr(value, 'item'):
#             value = value.item()

#         label = f"{prefix}/{label}"
#         self.writer.add_scalar(label, value, step)

#     def add_scalar_dict(self, scalar_dict, step, prefix=''):
#          for k, v in scalar_dict.items():
#              self.add_scalar(k, v, step, prefix=prefix)

#     @property
#     def experiment_dir(self):
#         return self._experiment_dir

#     @property
#     def experiment_name(self):
#         return self._experiment_name



class WandbLogger:
    def __init__(self, config, logdir, project_name, experiment_name):
        wandb.init(dir=logdir, project=project_name, name=experiment_name, config=config)
        self._experiment_dir = wandb.run.dir
        self._experiment_name = experiment_name

        # print(f"Experiment dir: {os.path.join(logdir, 'wandb', wandb.run.id)}")
        print(f"Experiment dir: {self._experiment_dir}")

        # save config
        self._add_config(config)

    def _add_config(self, config):
        config_path = os.path.join(self.experiment_dir, "config.yaml")
        with open(config_path, 'w') as f:
            OmegaConf.save(config, f, resolve=False)
            # OmegaConf.save(config, f, resolve=True)

    @staticmethod
    def _value_to_scalar(value):
        if hasattr(value, 'item'):
            value = value.item()
        
        return value

    @staticmethod
    def _build_label(key, mode="", scope="", prefix=""):
        label = ""
        label += f"{mode}/" if mode else ""
        label += f"{scope}/" if scope else ""
        label += f"{prefix}/" if prefix else ""

        label += key

        return label

    @staticmethod
    def _add_x_axis_values_to_log_dict(log_dict, mode="", epoch=None, n_batches_passed=None, n_samples_passed=None):
        def add_to_dict_if_not_none(d, key, value):
            if value is not None:
                d[key] = value

        add_to_dict_if_not_none(log_dict, f"epoch", epoch)
        add_to_dict_if_not_none(log_dict, f"{mode}/n_batches_passed", n_batches_passed)
        add_to_dict_if_not_none(log_dict, f"{mode}/n_samples_passed", n_samples_passed)

    def log_image(self, key, image, mode="", scope="", prefix="", epoch=None, n_batches_passed=None, n_samples_passed=None):
        log_dict = dict()

        # add image
        label = self._build_label(key, mode=mode, scope=scope, prefix=prefix)
        log_dict[label] = wandb.Image(image, caption=key)

        # add x-axis values
        self._add_x_axis_values_to_log_dict(log_dict, mode=mode, epoch=epoch, n_batches_passed=n_batches_passed, n_samples_passed=n_samples_passed)

        # log
        wandb.log(log_dict)

    def log_scalar(self, key, value, mode="", scope="", prefix="", epoch=None, n_batches_passed=None, n_samples_passed=None):
        log_dict = dict()

        # add value
        value = self._value_to_scalar(value)
        label = self._build_label(key, mode=mode, scope=scope, prefix=prefix)
        log_dict[label] = value

        # add x-axis values
        self._add_x_axis_values_to_log_dict(log_dict, mode=mode, epoch=epoch, n_batches_passed=n_batches_passed, n_samples_passed=n_samples_passed)

        # log
        wandb.log(log_dict)

    def log_scalar_dict(self, scalar_dict, mode="", scope="", prefix="", epoch=None, n_batches_passed=None, n_samples_passed=None):
        log_dict = dict()

        # add values from dict
        for key, value in scalar_dict.items():
            value = self._value_to_scalar(value)
            label = self._build_label(key, mode=mode, scope=scope, prefix=prefix)
            log_dict[label] = value

        # add x-axis values
        self._add_x_axis_values_to_log_dict(log_dict, mode=mode, epoch=epoch, n_batches_passed=n_batches_passed, n_samples_passed=n_samples_passed)
        
        # log
        wandb.log(log_dict)

    @property
    def experiment_dir(self):
        return self._experiment_dir

    @property
    def experiment_name(self):
        return self._experiment_name


def log_triple_smplx(logger, smplx_model, renderer, input_dict, output_dict, n_samples, mode="", scope="", prefix="", epoch=None, n_batches_passed=None, n_samples_passed=None):
    canvases = utils.vis.vis_triple_with_smplx(smplx_model, renderer, input_dict, output_dict, n_samples)
    canvas = np.concatenate(canvases, axis=0)

    logger.log_image(
        "triple_smplx", canvas,
        mode=mode, scope=scope, prefix=prefix,
        epoch=epoch, n_batches_passed=n_batches_passed, n_samples_passed=n_samples_passed
    )

def log_triple_smplx_keypoints_2d(logger, smplx_model, input_dict, output_dict, n_samples, mode="", scope="", prefix="", epoch=None, n_batches_passed=None, n_samples_passed=None):
    canvases = utils.vis.vis_triple_with_smplx_keypoints_2d(smplx_model, input_dict, output_dict, n_samples)
    canvas = np.concatenate(canvases, axis=0)

    logger.log_image(
        "triple_smplx_keypoints_2d", canvas,
        mode=mode, scope=scope, prefix=prefix,
        epoch=epoch, n_batches_passed=n_batches_passed, n_samples_passed=n_samples_passed
    )

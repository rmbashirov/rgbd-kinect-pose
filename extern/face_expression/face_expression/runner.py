import os
from tqdm import tqdm
import numpy as np
import pydoc

import torch
from torch import nn, optim

import face_expression
from face_expression import utils


class Runner(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # predictor
        predictor_cls = pydoc.locate(config.model.predictor.cls)
        self.predictor = predictor_cls(
            **({} if config.model.predictor.args is None else config.model.predictor.args)
        )

        # smplx model
        smplx_model_cls = pydoc.locate(config.model.smplx_model.cls)
        self.smplx_model = smplx_model_cls(
            comp_device=config.device,
            **({} if config.model.smplx_model.args is None else config.model.smplx_model.args)
        )

        # renderer
        self.renderer = None
        if self.config.log.render_smplx:
            renderer_cls = pydoc.locate(config.renderer.cls)
            self.renderer = renderer_cls(
                faces=self.smplx_model.f.cpu().numpy(),
                **({} if config.renderer.args is None else config.renderer.args)
            )

        # criterions and metrics
        ## expression criterion
        expression_criterion_cls = pydoc.locate(config.criterion.expression.cls)
        self.expression_criterion = expression_criterion_cls(**({} if config.criterion.expression.args is None else config.criterion.expression.args))

        ## jaw pose criterion
        jaw_pose_criterion_cls = pydoc.locate(config.criterion.jaw_pose.cls)
        self.jaw_pose_criterion = jaw_pose_criterion_cls(**({} if config.criterion.jaw_pose.args is None else config.criterion.jaw_pose.args))

        ## keypoint l2 criterion
        keypoint_2d_l2_criterion_cls = pydoc.locate(config.criterion.keypoint_2d_l2.cls)
        self.keypoint_2d_l2_criterion = keypoint_2d_l2_criterion_cls(**({} if config.criterion.keypoint_2d_l2.args is None else config.criterion.keypoint_2d_l2.args))

        ## keypoint l2 criterion
        keypoint_3d_l2_criterion_cls = pydoc.locate(config.criterion.keypoint_3d_l2.cls)
        self.keypoint_3d_l2_criterion = keypoint_3d_l2_criterion_cls(**({} if config.criterion.keypoint_3d_l2.args is None else config.criterion.keypoint_3d_l2.args))


        # init state
        self._init_state()

    def _init_state(self):
        self.state = {
            'epoch': 0,

            'train': {
                'n_batches_passed': 0,
                'n_samples_passed': 0
            },

            'val': {
                'n_batches_passed': 0,
                'n_samples_passed': 0
            }
        }

    def forward(self, input_dict):
        expression_pred, jaw_pose_pred = self.predictor.forward(input_dict['keypoints'], input_dict['beta'])

        # save outputs
        output_dict = dict()

        output_dict['expression_pred'] = expression_pred
        output_dict['jaw_pose_pred'] = jaw_pose_pred

        pose_pred = input_dict['pose'].clone()
        pose_pred[:, 63:66] = jaw_pose_pred
        output_dict['pose_pred'] = pose_pred

        return output_dict

    def get_state_dict(self, optimizer):
        state_dict = {
            'predictor': self.predictor.state_dict(),
            'optimizer': optimizer.state_dict(),

            'meta': {
                'epoch': self.state['epoch'],
                'n_batches_passed': self.state['train']['n_batches_passed'],
                'n_samples_passed': self.state['train']['n_samples_passed']
            }
        }

        return state_dict

    def load_state_dict(self, state_dict):
        self.predictor.load_state_dict(state_dict['predictor'])

    def get_optimizer(self):
        optimizer_cls = pydoc.locate(self.config.optimizer.predictor.cls)
        optimizer = optimizer_cls(
            self.predictor.parameters(),
            **({} if self.config.optimizer.predictor.args is None else self.config.optimizer.predictor.args)
        )

        return optimizer

    def run_epoch(self, dataloader, optimizer, logger, mode, n_iters_per_epoch=-1):
        assert mode in {'train', 'val'}

        if mode == 'train':
            torch.autograd.set_grad_enabled(True)
            self.predictor.train()
        elif mode == 'val':
            torch.autograd.set_grad_enabled(False)
            self.predictor.eval()

        # randomize distributed sampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(self.state['epoch'])

        n_iters_per_epoch = n_iters_per_epoch if n_iters_per_epoch > 0 else float('+inf')
        pbar = range(min(len(dataloader), n_iters_per_epoch))
        if self.config.local_rank == 0:
            pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
            pbar.set_description(f"{mode} | {self.config.log.experiment_name}")

        epoch_metrics = []
        dataloader_iter = iter(dataloader)
        for i in pbar:
            if i > n_iters_per_epoch:
                break

            batch_metrics = dict()
            output_dict = dict()

            # get batch
            input_dict = next(dataloader_iter)
            input_dict = utils.common.dict2device(input_dict, self.config.device, dtype=torch.float32)

            # forward
            output_dict = self.forward(input_dict)

            # loss
            ## expression loss
            expression_loss = self.expression_criterion(output_dict['expression_pred'], input_dict['expression'])
            batch_metrics['expression_loss'] = expression_loss.item()

            ## jaw pose loss
            jaw_pose_loss = self.jaw_pose_criterion(output_dict['jaw_pose_pred'], input_dict['pose'][:, 63:66])
            batch_metrics['jaw_pose_loss'] = jaw_pose_loss.item()

            ## keypoint l2 loss
            ### pred keypoints
            keypoints_3d_pred, _, _ = utils.misc.infer_smplx(
                self.smplx_model, output_dict['expression_pred'], output_dict['pose_pred'], input_dict['beta']
            )
            keypoints_2d_pred = utils.misc.project_keypoints_3d(keypoints_3d_pred, input_dict['projection_matrix'])

            ### target keypoints
            keypoints_3d_target, _, _ = utils.misc.infer_smplx(
                self.smplx_model, input_dict['expression'], input_dict['pose'], input_dict['beta']
            )
            keypoints_2d_target = utils.misc.project_keypoints_3d(keypoints_3d_target, input_dict['projection_matrix'])

            ### calculate keypoint losses
            keypoint_3d_l2_loss = self.keypoint_3d_l2_criterion(keypoints_3d_pred, keypoints_3d_target)
            batch_metrics['keypoint_3d_l2_loss'] = keypoint_3d_l2_loss.item()

            keypoint_2d_l2_loss = self.keypoint_2d_l2_criterion(keypoints_2d_pred, keypoints_2d_target)
            batch_metrics['keypoint_2d_l2_loss'] = keypoint_2d_l2_loss.item()

            ## total loss
            loss = \
                self.config.criterion.expression.weight * expression_loss + \
                self.config.criterion.jaw_pose.weight * jaw_pose_loss + \
                self.config.criterion.keypoint_3d_l2.weight * keypoint_3d_l2_loss + \
                self.config.criterion.keypoint_2d_l2.weight * keypoint_2d_l2_loss
            batch_metrics['total_loss'] = loss.item()

            # optimization step
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # metrics
            ## expression metric
            with torch.no_grad():
                expression_l1 = torch.mean(torch.abs(output_dict['expression_pred'] - input_dict['expression']))
                batch_metrics['expression_l1'] = expression_l1.item()

            ## keypoint 3d l2 metric
            with torch.no_grad():
                keypoint_3d_l2_loss = self.keypoint_3d_l2_criterion(keypoints_3d_pred, keypoints_3d_target)
                batch_metrics['keypoint_3d_l2_loss'] = keypoint_3d_l2_loss.item()

            # collect metrics
            epoch_metrics.append(batch_metrics)

            batch_metrics = utils.distributed.reduce_loss_dict(batch_metrics)
            batch_metrics = utils.common.squeeze_metrics(batch_metrics)

            # log per-batch metrics
            if self.config.local_rank == 0:
                # log batch scalar metrics
                if mode == 'train':
                    logger.log_scalar_dict(
                        batch_metrics,
                        mode=mode, scope="batch", prefix="",
                        epoch=self.state['epoch'], n_batches_passed=self.state[mode]['n_batches_passed'], n_samples_passed=self.state[mode]['n_samples_passed']
                    )

                if self.state[mode]['n_batches_passed'] % self.config.log.log_freq_image_batch[mode] == 0:
                    if self.config.log.render_smplx:
                        utils.logger.log_triple_smplx(
                            logger,
                            self.smplx_model, self.renderer,
                            input_dict, output_dict,
                            self.config.log.log_n_samples_triple,
                            mode=mode, scope="batch", prefix="",
                            epoch=self.state['epoch'], n_batches_passed=self.state[mode]['n_batches_passed'], n_samples_passed=self.state[mode]['n_samples_passed']
                        )
                    else:
                        utils.logger.log_triple_smplx_keypoints_2d(
                            logger,
                            self.smplx_model,
                            input_dict, output_dict,
                            self.config.log.log_n_samples_triple,
                            mode=mode, scope="batch", prefix="",
                            epoch=self.state['epoch'], n_batches_passed=self.state[mode]['n_batches_passed'], n_samples_passed=self.state[mode]['n_samples_passed']
                        )

            # update state
            if self.config.local_rank == 0:
                self.state[mode]['n_batches_passed'] += 1
                self.state[mode]['n_samples_passed'] += len(input_dict['image'])

        # log per-epoch metrics
        if self.config.local_rank == 0:
            epoch_metrics = utils.common.reduce_metrics(epoch_metrics)
            logger.log_scalar_dict(
                epoch_metrics,
                mode=mode, scope="epoch", prefix="",
                epoch=self.state['epoch'], n_batches_passed=self.state[mode]['n_batches_passed'], n_samples_passed=self.state[mode]['n_samples_passed']
            )

            # save checkpoint
            if mode == 'train' and self.state['epoch'] % self.config.log.log_freq_checkpoint_epoch == 0:
                checkpoint_dir = os.path.join(logger.experiment_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                    
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_{:06}.pth".format(self.state['epoch']))
                torch.save(
                    self.get_state_dict(optimizer),
                    checkpoint_path
                )


class RunnerMouth(nn.Module):
    SMPLX_MOUTH_INDICES = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]

    def __init__(self, config):
        super().__init__()

        self.config = config

        # predictor
        predictor_cls = pydoc.locate(config.model.predictor.cls)
        self.predictor = predictor_cls(
            **({} if config.model.predictor.args is None else config.model.predictor.args)
        )

        # smplx model
        smplx_model_cls = pydoc.locate(config.model.smplx_model.cls)
        self.smplx_model = smplx_model_cls(
            comp_device=config.device,
            **({} if config.model.smplx_model.args is None else config.model.smplx_model.args)
        )

        # renderer
        self.renderer = None
        if self.config.log.render_smplx:
            renderer_cls = pydoc.locate(config.renderer.cls)
            self.renderer = renderer_cls(
                faces=self.smplx_model.f.cpu().numpy(),
                **({} if config.renderer.args is None else config.renderer.args)
            )

        # criterions and metrics
        ## expression criterion
        expression_criterion_cls = pydoc.locate(config.criterion.expression.cls)
        self.expression_criterion = expression_criterion_cls(**({} if config.criterion.expression.args is None else config.criterion.expression.args))

        ## jaw pose criterion
        jaw_pose_criterion_cls = pydoc.locate(config.criterion.jaw_pose.cls)
        self.jaw_pose_criterion = jaw_pose_criterion_cls(**({} if config.criterion.jaw_pose.args is None else config.criterion.jaw_pose.args))

        ## keypoint 2d l2 criterion
        keypoint_2d_l2_criterion_cls = pydoc.locate(config.criterion.keypoint_2d_l2.cls)
        self.keypoint_2d_l2_criterion = keypoint_2d_l2_criterion_cls(**({} if config.criterion.keypoint_2d_l2.args is None else config.criterion.keypoint_2d_l2.args))

        ## keypoint 3d l2 criterion
        keypoint_3d_l2_criterion_cls = pydoc.locate(config.criterion.keypoint_3d_l2.cls)
        self.keypoint_3d_l2_criterion = keypoint_3d_l2_criterion_cls(**({} if config.criterion.keypoint_3d_l2.args is None else config.criterion.keypoint_3d_l2.args))

        ## keypoint 3d mouth l2 criterion
        keypoint_3d_mouth_l2_criterion_cls = pydoc.locate(config.criterion.keypoint_3d_mouth_l2.cls)
        self.keypoint_3d_mouth_l2_criterion = keypoint_3d_mouth_l2_criterion_cls(**({} if config.criterion.keypoint_3d_mouth_l2.args is None else config.criterion.keypoint_3d_mouth_l2.args))

        # init state
        self._init_state()

    def _init_state(self):
        self.state = {
            'epoch': 0,

            'train': {
                'n_batches_passed': 0,
                'n_samples_passed': 0
            },

            'val': {
                'n_batches_passed': 0,
                'n_samples_passed': 0
            }
        }

    def forward(self, input_dict):
        expression_pred, jaw_pose_pred = self.predictor.forward(input_dict['keypoints'], input_dict['beta'])

        # save outputs
        output_dict = dict()

        output_dict['expression_pred'] = expression_pred
        output_dict['jaw_pose_pred'] = jaw_pose_pred

        pose_pred = input_dict['pose'].clone()
        pose_pred[:, 63:66] = jaw_pose_pred
        output_dict['pose_pred'] = pose_pred

        return output_dict

    def get_state_dict(self, optimizer):
        state_dict = {
            'predictor': self.predictor.state_dict(),
            'optimizer': optimizer.state_dict(),

            'meta': {
                'epoch': self.state['epoch'],
                'n_batches_passed': self.state['train']['n_batches_passed'],
                'n_samples_passed': self.state['train']['n_samples_passed']
            }
        }

        return state_dict

    def load_state_dict(self, state_dict):
        self.predictor.load_state_dict(state_dict['predictor'])

    def get_optimizer(self):
        optimizer_cls = pydoc.locate(self.config.optimizer.predictor.cls)
        optimizer = optimizer_cls(
            self.predictor.parameters(),
            **({} if self.config.optimizer.predictor.args is None else self.config.optimizer.predictor.args)
        )

        return optimizer

    def run_epoch(self, dataloader, optimizer, logger, mode, n_iters_per_epoch=-1):
        assert mode in {'train', 'val'}

        if mode == 'train':
            torch.autograd.set_grad_enabled(True)
            self.predictor.train()
        elif mode == 'val':
            torch.autograd.set_grad_enabled(False)
            self.predictor.eval()

        # randomize distributed sampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(self.state['epoch'])

        n_iters_per_epoch = n_iters_per_epoch if n_iters_per_epoch > 0 else float('+inf')
        pbar = range(min(len(dataloader), n_iters_per_epoch))
        if self.config.local_rank == 0:
            pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)
            pbar.set_description(f"{mode} | {self.config.log.experiment_name}")

        epoch_metrics = []
        dataloader_iter = iter(dataloader)
        for i in pbar:
            if i > n_iters_per_epoch:
                break

            batch_metrics = dict()
            output_dict = dict()

            # get batch
            input_dict = next(dataloader_iter)
            input_dict = utils.common.dict2device(input_dict, self.config.device, dtype=torch.float32)

            # forward
            output_dict = self.forward(input_dict)

            # loss
            ## expression loss
            expression_loss = self.expression_criterion(output_dict['expression_pred'], input_dict['expression'])
            batch_metrics['expression_loss'] = expression_loss.item()

            ## jaw pose loss
            jaw_pose_loss = self.jaw_pose_criterion(output_dict['jaw_pose_pred'], input_dict['pose'][:, 63:66])
            batch_metrics['jaw_pose_loss'] = jaw_pose_loss.item()

            ## keypoint l2 loss
            ### pred keypoints
            keypoints_3d_pred, _, _ = utils.misc.infer_smplx(
                self.smplx_model, output_dict['expression_pred'], output_dict['pose_pred'], input_dict['beta']
            )
            keypoints_2d_pred = utils.misc.project_keypoints_3d(keypoints_3d_pred, input_dict['projection_matrix'])

            ### target keypoints
            keypoints_3d_target, _, _ = utils.misc.infer_smplx(
                self.smplx_model, input_dict['expression'], input_dict['pose'], input_dict['beta']
            )
            keypoints_2d_target = utils.misc.project_keypoints_3d(keypoints_3d_target, input_dict['projection_matrix'])

            ### calculate keypoint losses
            keypoint_3d_l2_loss = self.keypoint_3d_l2_criterion(keypoints_3d_pred, keypoints_3d_target)
            batch_metrics['keypoint_3d_l2_loss'] = keypoint_3d_l2_loss.item()

            keypoint_2d_l2_loss = self.keypoint_2d_l2_criterion(keypoints_2d_pred, keypoints_2d_target)
            batch_metrics['keypoint_2d_l2_loss'] = keypoint_2d_l2_loss.item()

            keypoint_3d_mouth_l2_loss = self.keypoint_3d_mouth_l2_criterion(
                keypoints_3d_pred[:, self.SMPLX_MOUTH_INDICES],
                keypoints_3d_target[:, self.SMPLX_MOUTH_INDICES]
            )
            batch_metrics['keypoint_3d_mouth_l2_loss'] = keypoint_3d_mouth_l2_loss.item()

            ## total loss
            loss = \
                self.config.criterion.expression.weight * expression_loss + \
                self.config.criterion.jaw_pose.weight * jaw_pose_loss + \
                self.config.criterion.keypoint_3d_l2.weight * keypoint_3d_l2_loss + \
                self.config.criterion.keypoint_2d_l2.weight * keypoint_2d_l2_loss + \
                self.config.criterion.keypoint_3d_mouth_l2.weight * keypoint_3d_mouth_l2_loss
            batch_metrics['total_loss'] = loss.item()

            # optimization step
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # metrics
            ## expression metric
            with torch.no_grad():
                expression_l1 = torch.mean(torch.abs(output_dict['expression_pred'] - input_dict['expression']))
                batch_metrics['expression_l1'] = expression_l1.item()

            ## keypoint 3d l2 metric
            with torch.no_grad():
                keypoint_3d_l2_loss = self.keypoint_3d_l2_criterion(keypoints_3d_pred, keypoints_3d_target)
                batch_metrics['keypoint_3d_l2_loss'] = keypoint_3d_l2_loss.item()

            # collect metrics
            epoch_metrics.append(batch_metrics)

            batch_metrics = utils.distributed.reduce_loss_dict(batch_metrics)
            batch_metrics = utils.common.squeeze_metrics(batch_metrics)

            # log per-batch metrics
            if self.config.local_rank == 0:
                # log batch scalar metrics
                if mode == 'train':
                    logger.log_scalar_dict(
                        batch_metrics,
                        mode=mode, scope="batch", prefix="",
                        epoch=self.state['epoch'], n_batches_passed=self.state[mode]['n_batches_passed'], n_samples_passed=self.state[mode]['n_samples_passed']
                    )

                if self.state[mode]['n_batches_passed'] % self.config.log.log_freq_image_batch[mode] == 0:
                    if self.config.log.render_smplx:
                        utils.logger.log_triple_smplx(
                            logger,
                            self.smplx_model, self.renderer,
                            input_dict, output_dict,
                            self.config.log.log_n_samples_triple,
                            mode=mode, scope="batch", prefix="",
                            epoch=self.state['epoch'], n_batches_passed=self.state[mode]['n_batches_passed'], n_samples_passed=self.state[mode]['n_samples_passed']
                        )
                    else:
                        utils.logger.log_triple_smplx_keypoints_2d(
                            logger,
                            self.smplx_model,
                            input_dict, output_dict,
                            self.config.log.log_n_samples_triple,
                            mode=mode, scope="batch", prefix="",
                            epoch=self.state['epoch'], n_batches_passed=self.state[mode]['n_batches_passed'], n_samples_passed=self.state[mode]['n_samples_passed']
                        )

            # update state
            if self.config.local_rank == 0:
                self.state[mode]['n_batches_passed'] += 1
                self.state[mode]['n_samples_passed'] += len(input_dict['image'])

        # log per-epoch metrics
        if self.config.local_rank == 0:
            epoch_metrics = utils.common.reduce_metrics(epoch_metrics)
            logger.log_scalar_dict(
                epoch_metrics,
                mode=mode, scope="epoch", prefix="",
                epoch=self.state['epoch'], n_batches_passed=self.state[mode]['n_batches_passed'], n_samples_passed=self.state[mode]['n_samples_passed']
            )

            # save checkpoint
            if mode == 'train' and self.state['epoch'] % self.config.log.log_freq_checkpoint_epoch == 0:
                checkpoint_dir = os.path.join(logger.experiment_dir, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                    
                checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_{:06}.pth".format(self.state['epoch']))
                torch.save(
                    self.get_state_dict(optimizer),
                    checkpoint_path
                )
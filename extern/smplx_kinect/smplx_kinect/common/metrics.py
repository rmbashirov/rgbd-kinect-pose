import numpy as np
import cv2
from copy import deepcopy
import operator

from smplx_kinect.common.angle_representation import universe_convert
from smplx_kinect.common.angle_representation import rotmat2euler
from smplx_kinect.common.angle_representation import is_valid_rotmat, local_rot_to_global, get_closest_rotmat


def pck(predictions, targets, thresh):
    """
    Percentage of correct keypoints.
    Args:
        predictions: np array of predicted 3D joint positions in format (..., n_joints, 3)
        targets: np array of same shape as `predictions`
        thresh: radius within which a predicted joint has to lie.

    Returns:
        Percentage of correct keypoints at the given threshold level, stored in a np array of shape (..., len(threshs))

    """
    dist = np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))
    pck = np.mean(np.array(dist <= thresh, dtype=np.float32), axis=-1)
    return pck


def angle_diff(predictions, targets):
    """
    Computes the angular distance between the target and predicted rotations. We define this as the angle that is
    required to rotate one rotation into the other. This essentially computes || log(R_diff) || where R_diff is the
    difference rotation between prediction and target.

    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The geodesic distance for each joint as an np array of shape (..., n_joints)
    """
    assert predictions.shape[-1] == predictions.shape[-2] == 3
    assert targets.shape[-1] == targets.shape[-2] == 3

    ori_shape = predictions.shape[:-2]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = np.matmul(preds, np.transpose(targs, [0, 2, 1]))

    # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
    # the predicted and target orientations
    angles = []
    for i in range(r.shape[0]):
        aa, _ = cv2.Rodrigues(r[i])
        angles.append(np.linalg.norm(aa))
    angles = np.array(angles)

    return np.reshape(angles, ori_shape)


def positional(predictions, targets):
    """
    Computes the Euclidean distance between joints in 3D space.
    Args:
        predictions: np array of predicted 3D joint positions in format (..., n_joints, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The Euclidean distance for each joint as an np array of shape (..., n_joints)
    """
    return np.sqrt(np.sum((predictions - targets) ** 2, axis=-1))


def euler_diff(predictions, targets):
    """
    Computes the Euler angle error as in previous work, following
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/translate.py#L207
    Args:
        predictions: np array of predicted joint angles represented as rotation matrices, i.e. in shape
          (..., n_joints, 3, 3)
        targets: np array of same shape as `predictions`

    Returns:
        The Euler angle error an np array of shape (..., )
    """
    assert predictions.shape[-1] == 3 and predictions.shape[-2] == 3
    assert targets.shape[-1] == 3 and targets.shape[-2] == 3
    n_joints = predictions.shape[-3]

    #print(predictions.shape[:-2])
    ori_shape = predictions.shape[:-3]
    preds = np.reshape(predictions, [-1, 3, 3])
    targs = np.reshape(targets, [-1, 3, 3])

    euler_preds = rotmat2euler(preds)  # (N, 3)
    euler_targs = rotmat2euler(targs)  # (N, 3)

    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = np.reshape(euler_preds, [-1, n_joints*3])
    euler_targs = np.reshape(euler_targs, [-1, n_joints*3])

    # l2 error on euler angles
    #print(euler_preds.shape,euler_targs.shape)
    if euler_targs.shape[0] == 1:
        idx_to_use = np.arange(euler_targs.shape[1])
    else:
        idx_to_use = np.where(np.std(euler_targs, 0) > 1e-4)[0]

    euc_error = np.power(euler_targs[:,idx_to_use] - euler_preds[:,idx_to_use], 2)
    euc_error = np.sqrt(np.sum(euc_error, axis=1))  # (-1, ...)

    # reshape to original
    return np.reshape(euc_error, ori_shape)


class MetricsEngine:
    def __init__(self, smplx_wrapper, which=None, pck_thresholds=None, force_valid_rot=True, is_jaw=False):
        self.force_valid_rot = force_valid_rot

        if which is None:
            which = ["positional", "joint_angle", "euler"]
        self.which = which

        if pck_thresholds is None:
            pck_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
        self.pck_thresholds = pck_thresholds
        self.pck_thresholds_str = [f'pck_{pck_threshold}' for pck_threshold in pck_thresholds]
        self.which += self.pck_thresholds_str

        self.dof = 9
        self.left_mult = False
        self.smplx_wrapper = smplx_wrapper
        self.body_model = smplx_wrapper.models['male']
        self.n_joints = self.body_model.NUM_BODY_JOINTS + 1
        self.is_jaw = is_jaw
        if is_jaw:
            self.n_joints += 1
        self.parents = self.body_model.parents[:self.n_joints].cpu().numpy()

        self.metrics_agg = {k: None for k in self.which}
        self.n_samples = 0

    def from_rotmat(self, pose_all):
        if self.is_jaw:
            pose = pose_all.reshape(-1, self.n_joints * 9)[:, :-9]
            jaw_pose = pose_all.reshape(-1, 1 * 9)[:, -9:]
        else:
            pose = pose_all.reshape(-1, self.n_joints * 9)
            jaw_pose = None
        B = pose.shape[0]
        result = np.zeros((B, self.n_joints, 3))
        for b in range(B):
            if jaw_pose is not None:
                jaw_pose = universe_convert(jaw_pose[b, :], 'rotmtx', 'aa', multiple=True)
            model_output = self.smplx_wrapper.get_output(
                gender='male',
                betas=np.zeros(10),
                body_pose=universe_convert(pose[b, 9:], 'rotmtx', 'aa', multiple=True),
                rvec=universe_convert(pose[b, :9], 'rotmtx', 'aa', multiple=True),
                tvec=np.zeros(3),
                jaw_pose=jaw_pose
            )
            joints = model_output.joints.detach().cpu().numpy()[0, :self.n_joints]
            result[b] = joints
        return result

    def reset(self):
        """
        Reset all metrics.
        """
        self.metrics_agg = {k: None for k in self.which}
        self.n_samples = 0

    def compute(self, predictions, targets, reduce_fn="mean"):
        """
        Compute the chosen metrics. Predictions and targets are assumed to be in rotation matrix format.
        Args:
            predictions: An np array of shape (batch_size, seq_length, *9)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].

        Returns:
            A dictionary {metric_name -> values} where the values are given per batch entry and frame as an np array
            of shape (n, seq_length). `reduce_fn` is only applied to metrics where it makes sense, i.e. not to PCK
            and euler angle differences.
        """
        assert len(predictions.shape) == len(targets.shape), f'{predictions.shape} != {targets.shape}'
        for i in range(len(predictions.shape)):
            assert predictions.shape[i] == targets.shape[i], f'{predictions.shape} != {targets.shape}'
        assert predictions.shape[-1] % self.dof == 0, "predictions are not rotation matrices"
        assert targets.shape[-1] % self.dof == 0, "targets are not rotation matrices"
        n_joints = int(predictions.shape[-1] / self.dof)
        assert n_joints == self.n_joints

        batch_size = predictions.shape[0]
        seq_length = predictions.shape[1]

        pred = np.reshape(predictions, [-1, n_joints*self.dof]).copy()
        targ = np.reshape(targets, [-1, n_joints*self.dof]).copy()

        # enforce valid rotations
        if self.force_valid_rot:
            pred_val = np.reshape(pred, [-1, n_joints, 3, 3])
            pred = get_closest_rotmat(pred_val)
            pred = np.reshape(pred, [-1, n_joints*self.dof])

        # check that the rotations are valid
        pred_are_valid = is_valid_rotmat(np.reshape(pred, [-1, n_joints, 3, 3]))
        assert pred_are_valid, 'predicted rotation matrices are not valid'
        targ_are_valid = is_valid_rotmat(np.reshape(targ, [-1, n_joints, 3, 3]), thresh=1e-5)
        assert targ_are_valid, 'target rotation matrices are not valid'

        # make sure we don't consider the root orientation
        pred[:, 0:9] = np.eye(3, 3).flatten()
        targ[:, 0:9] = np.eye(3, 3).flatten()

        metrics = dict()

        pred_pos = self.from_rotmat(pred)  # (-1, full_n_joints, 3)
        targ_pos = self.from_rotmat(targ)  # (-1, full_n_joints, 3)

        reduce_fn_np = np.mean if reduce_fn == "mean" else np.sum

        select_joints = range(n_joints)

        for metric in self.which:
            if metric.startswith("pck"):
                thresh = float(metric.split("_")[-1])
                v = pck(pred_pos[:, select_joints], targ_pos[:, select_joints], thresh=thresh)  # (-1, )
                metrics[metric] = np.reshape(v, [batch_size, seq_length])
            elif metric == "positional":
                v = positional(pred_pos[:, select_joints], targ_pos[:, select_joints])  # (-1, n_joints)
                v = np.reshape(v, [batch_size, seq_length, n_joints])
                metrics[metric] = reduce_fn_np(v, axis=-1)
            elif metric == "joint_angle":
                # compute the joint angle diff on the global rotations, not the local ones, which is a harder metric
                pred_global = local_rot_to_global(pred, self.parents, left_mult=self.left_mult,
                                                  rep="rotmat")  # (-1, full_n_joints, 3, 3)
                targ_global = local_rot_to_global(targ, self.parents, left_mult=self.left_mult,
                                                  rep="rotmat")  # (-1, full_n_joints, 3, 3)
                v = angle_diff(pred_global[:, select_joints], targ_global[:, select_joints])  # (-1, n_joints)
                v = np.reshape(v, [batch_size, seq_length, n_joints])
                metrics[metric] = reduce_fn_np(v, axis=-1)
            elif metric == "euler":
                # compute the euler angle error on the local rotations, which is how previous work does it
                pred_local = np.reshape(pred, [-1, n_joints, 3, 3])
                targ_local = np.reshape(targ, [-1, n_joints, 3, 3])
                v = euler_diff(pred_local[:, select_joints], targ_local[:, select_joints])  # (-1, )
                metrics[metric] = np.reshape(v, [batch_size, seq_length])
            else:
                raise ValueError("metric '{}' unknown".format(metric))

        return metrics

    def aggregate(self, metrics):
        """
        Aggregate the metrics.
        Args:
            metrics: Dictionary of new metric values to aggregate. Each entry is expected to be a numpy array
            of shape (batch_size, seq_length). For PCK values there might be more than 2 dimensions.
        """
        assert isinstance(metrics, dict)
        assert list(metrics.keys()) == list(self.metrics_agg.keys())

        # sum over the batch dimension
        for m in metrics:
            if self.metrics_agg[m] is None:
                self.metrics_agg[m] = np.sum(metrics[m], axis=0)
            else:
                self.metrics_agg[m] += np.sum(metrics[m], axis=0)

        # keep track of the total number of samples processed
        batch_size = metrics[list(metrics.keys())[0]].shape[0]
        self.n_samples += batch_size

    def compute_and_aggregate(self, predictions, targets, reduce_fn="mean"):
        """
        Computes the metric values and aggregates them directly.
        Args:
            predictions: An np array of shape (n, seq_length, n_joints*dof)
            targets: An np array of the same shape as `predictions`
            reduce_fn: Which reduce function to apply to the joint dimension, if applicable. Choices are [mean, sum].
        """
        metrics = self.compute(predictions, targets, reduce_fn)
        self.aggregate(metrics)

    def get_summary_metrics(self, target_lengths, at_mode=True):
        """
        target_lengths: Metrics at these time-steps are reported.
        at_mode: If true will report the numbers at the last frame rather then until the last frame.
        """

        assert self.n_samples > 0

        summary_metrics = dict()
        for m in self.metrics_agg:
            summary_metrics[m] = self.metrics_agg[m] / self.n_samples

        result = dict()
        for seq_length in sorted(target_lengths):
            seq_length_result = dict()
            for k, v in sorted(summary_metrics.items(), key=operator.itemgetter(0)):
                val = v[seq_length - 1] if at_mode else np.sum(v[:seq_length])
                seq_length_result[k] = val
            seq_length_result['auc'] = self.calculate_auc(seq_length_result)
            result[seq_length] = seq_length_result
        return result

    def get_summary_metrics_str(self, summary_metrics):
        s = ""
        for seq_length, seq_length_result in sorted(summary_metrics.items(), key=operator.itemgetter(0)):
            s += "\nMetrics until {:<2}:".format(seq_length)
            for k, v in sorted(seq_length_result.items(), key=operator.itemgetter(0)):
                s += "   {}: {:.3f}".format(k, v)
        return s

    def calculate_auc(self, summary_metrics):
        norm_factor = np.diff(self.pck_thresholds).sum()
        auc_values = []

        for i in range(len(self.pck_thresholds) - 1):
            auc = (summary_metrics[self.pck_thresholds_str[i]] + summary_metrics[self.pck_thresholds_str[i + 1]]) / 2 \
                  * (self.pck_thresholds[i + 1] - self.pck_thresholds[i])
            auc_values.append(auc)
        return np.array(auc_values).sum() / norm_factor

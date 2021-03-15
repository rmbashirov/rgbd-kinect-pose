import os
import os.path as osp
import numpy as np

from copy import deepcopy
from collections import defaultdict

from smplx_kinect.common.angle_representation import universe_convert, local_rot_to_global, get_closest_rotmat, global_rot_to_local
from smplx_kinect.common.body_models import HandsSMPLXWrapper


mano_parents = [
    -1,
    0, 1, 2,
    0, 4, 5,
    0, 7, 8,
    0, 10, 11,
    0, 13, 14,
    3, 6, 9, 12, 15
][:16]


def process_mano_quats_impl(quats, flip):
    global_rotmtx = universe_convert(quats, 'quat', 'rotmtx')
    if flip:
        neg = np.eye(3)
        neg[0, 0] = -1
        for joint_index in range(len(global_rotmtx)):
            global_rotmtx[joint_index] = neg @ global_rotmtx[joint_index] @ neg
    local_rotmtx = global_rot_to_local(
        global_rotmtx,
        mano_parents,
        left_mult=False
    )
    local_aa = universe_convert(local_rotmtx, 'rotmtx', 'aa')
    return global_rotmtx, local_rotmtx, local_aa


def process_mano_quats(d):
    result = dict()
    for side, quats in d.items():
        quats = deepcopy(quats[:, [1, 2, 3, 0]])
        # minimal-hand provides 4 joints for each finger, select last 3
        quats = quats[[
            0,
            2, 3, 16,
            5, 6, 17,
            8, 9, 18,
            11, 12, 19,
            14, 15, 20
        ]]
        global_rotmtx, local_rotmtx, local_aa = process_mano_quats_impl(quats, flip=(side == 'right'))
        result[side] = {
            'global_rotmtx': global_rotmtx,
            'local_rotmtx': local_rotmtx,
            'local_aa': local_aa
        }
        for k in result[side]:
            result[side][k] = np.expand_dims(result[side][k], 0)
    return result


class Filter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev = None

    def add(self, cur):
        if self.prev is None:
            self.prev = cur
        else:
            self.prev = self.alpha * cur + (1 - self.alpha) * self.prev
        return self.prev


class RotMtxFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.prev = None

    def add(self, cur):
        cur = cur.reshape(-1, 3)
        cur = universe_convert(cur, 'aa', 'rotmtx')

        if self.prev is None:
            self.prev = cur
        else:
            self.prev = self.alpha * cur + (1 - self.alpha) * self.prev

        self.prev = get_closest_rotmat(self.prev)
        result = universe_convert(self.prev, 'rotmtx', 'aa')
        result = result.reshape(-1)
        return result


def get_local_joint(joint_index, joint_global_rotmtx, global_pose_rotmtx, parents):
    parent_global_rotmtx = global_pose_rotmtx[parents[joint_index]]
    joint_local_rotmtx = parent_global_rotmtx.T.dot(joint_global_rotmtx)
    joint_local_aa = universe_convert(joint_local_rotmtx, 'rotmtx', 'aa')
    # local_pose_aa[joint_index] = joint_local_rot
    return joint_local_rotmtx, joint_local_aa


class Filterer:
    def __init__(
        self,
        device, body_models_dp,
        modify_wrist=True, modify_wrist_th=0, modify_wrist_N=0, filter_wrist=True,
        filter_hand=True,
        filter_global_trans=True, filter_global_rot=True, filter_body_pose=True,
        filter_jaw_pose=True, filter_face_expression=True,
        fix_global=False
    ):
        self.device = device
        self.body_models_dp = body_models_dp
        self.modify_wrist = modify_wrist
        self.modify_wrist_th = modify_wrist_th
        self.modify_wrist_N = modify_wrist_N
        self.filter_wrist = filter_wrist
        self.filter_hand = filter_hand
        self.filter_global_trans = filter_global_trans
        self.filter_global_rot = filter_global_rot
        self.filter_body_pose = filter_body_pose
        self.filter_jaw_pose = filter_jaw_pose
        self.filter_face_expression = filter_face_expression
        self.fix_global = fix_global

        non_flat_smplx_wrapper = HandsSMPLXWrapper(
            self.body_models_dp, use_pca=False, device=self.device, flat_hand_mean=False)
        self.left_hand_mean = non_flat_smplx_wrapper.models['male'].left_hand_mean.cpu().numpy()
        self.right_hand_mean = non_flat_smplx_wrapper.models['male'].right_hand_mean.cpu().numpy()

        self.left_hand_filter = Filter(alpha=0.15)
        self.right_hand_filter = Filter(alpha=0.15)

        if modify_wrist:
            self.smplx_parents = deepcopy(non_flat_smplx_wrapper.models['male'].parents[:22])
            self.smplx_hand_joints = [
                [21, 'right_wrist'],
                [20, 'left_wrist']
            ]
            self.left_wrist_filter = Filter(alpha=0.3)
            self.right_wrist_filter = Filter(alpha=0.3)

        self.global_rot, self.global_trans = None, None
        self.global_trans_filter = Filter(alpha=0.25)
        self.body_pose_filter = RotMtxFilter(alpha=0.3)
        self.body_root_filter = RotMtxFilter(alpha=0.3)
        self.jaw_filter = RotMtxFilter(alpha=0.3)
        self.face_expression_filter = Filter(alpha=0.25)

        self.prev_hand_kinect_joints = None
        self.hand_kinect_diffs = defaultdict(list)
        # self.hist = []

    def filter(self, x):
        is_hand_pose = 'hand_pose' in x
        if is_hand_pose:
            hands_dict = x['hand_pose']
            hands_processed = process_mano_quats(hands_dict)

            inf_hand_poses = dict()
            inf_hand_root = dict()
            for hand_side in ['left', 'right']:
                hand_sample = hands_processed[hand_side]
                hand_pose_aa = hand_sample['local_aa'][0][1:].reshape(-1)

                inf_hand_poses[hand_side] = hand_pose_aa
                inf_hand_root[hand_side] = hand_sample['global_rotmtx'][0][0]

        kinect_joints = x['body_pose']['input_kinect_joints']

        # 'global_rot': body_pose['global_rot'],
        # 'global_trans': body_pose['global_trans'],
        body_pose = x['body_pose']['body_pose']
        if self.fix_global and self.global_rot is not None:
            pass
        else:
            self.global_rot = x['body_pose']['global_rot']
            # rvec_addon = np.array([0, np.pi / 2, 0], dtype=np.float32)
            # rvec_addon = universe_convert(rvec_addon, 'aa', 'rotmtx')
            #
            # result_rotmtx = universe_convert(self.global_rot, 'aa', 'rotmtx')
            # result_rotmtx = rvec_addon.dot(result_rotmtx)
            # self.global_rot = universe_convert(result_rotmtx, 'rotmtx', 'aa')

        # is_close_hands = x['body_pose']['is_close_hands']
        is_close_hands = False
        is_reset_hands = {
            'left': False,
            'right': False
        }
        if self.modify_wrist and self.modify_wrist_th > 0 and self.modify_wrist_N > 0:
            cur_hand_kinect_joints = {
                'left': kinect_joints[8],
                'right': kinect_joints[15]
            }
            if self.prev_hand_kinect_joints is not None:
                for joint_index, joint_name in self.smplx_hand_joints:
                    side = joint_name.split('_')[0]
                    diff = np.sqrt(np.sum((cur_hand_kinect_joints[side] - self.prev_hand_kinect_joints[side]) ** 2))
                    self.hand_kinect_diffs[side].append(diff)
                    if len(self.hand_kinect_diffs[side]) > self.modify_wrist_N:
                        self.hand_kinect_diffs[side].pop(0)
                        if np.mean(self.hand_kinect_diffs[side]) > self.modify_wrist_th:
                            is_reset_hands[side] = True
                            print(f'reset {side}')
            self.prev_hand_kinect_joints = cur_hand_kinect_joints

        if self.modify_wrist and not is_close_hands and is_hand_pose:
            body_pose = np.concatenate((self.global_rot, body_pose))
            body_pose = body_pose.reshape(-1, 3)
            local_body_pose_rotmtx = universe_convert(body_pose, 'aa', 'rotmtx')
            global_body_pose_rotmtx = local_rot_to_global(
                local_body_pose_rotmtx, self.smplx_parents)[0]

            for joint_index, joint_name in self.smplx_hand_joints:
                side = joint_name.split('_')[0]

                if is_reset_hands[side]:
                    continue

                hand_root_global_rotmtx = inf_hand_root[side]
                if self.filter_wrist:
                    root_filter = self.left_wrist_filter \
                        if joint_name.startswith('left') else self.right_wrist_filter
                    hand_root_global_rotmtx = root_filter.add(hand_root_global_rotmtx)
                    hand_root_global_rotmtx = get_closest_rotmat(hand_root_global_rotmtx[np.newaxis, :, :])[0]
                    root_filter.prev = hand_root_global_rotmtx
                joint_local_rotmtx, joint_local_aa = get_local_joint(
                    joint_index, hand_root_global_rotmtx,
                    global_body_pose_rotmtx, self.smplx_parents)

                body_pose[joint_index] = deepcopy(joint_local_aa)
            body_pose = body_pose[1:].reshape(-1)

        left_hand_pose, right_hand_pose = None, None
        if is_hand_pose:
            if not is_close_hands:
                if not is_reset_hands['left']:
                    left_hand_pose = inf_hand_poses['left']
                if not is_reset_hands['right']:
                    right_hand_pose = inf_hand_poses['right']
        if left_hand_pose is None:
            left_hand_pose = deepcopy(self.left_hand_mean)
        if right_hand_pose is None:
            right_hand_pose = deepcopy(self.right_hand_mean)
        if self.filter_hand:
            left_hand_pose = self.left_hand_filter.add(left_hand_pose)
            right_hand_pose = self.right_hand_filter.add(right_hand_pose)

        if self.filter_body_pose:
            body_pose = self.body_pose_filter.add(body_pose)

        if self.filter_global_rot:
            self.global_rot = self.body_root_filter.add(self.global_rot)
        if self.fix_global and self.global_trans is not None:
            pass
        else:
            self.global_trans = x['body_pose']['global_trans']
        if self.filter_global_trans:
            self.global_trans = self.global_trans_filter.add(self.global_trans)

        is_face_pose = 'face_pose' in x
        if is_face_pose:
            jaw_pose = x['face_pose']['jaw_pose']
            if self.filter_jaw_pose:
                jaw_pose = self.jaw_filter.add(jaw_pose)

            face_expression = x['face_pose']['expression']
            if self.filter_face_expression:
                face_expression = self.face_expression_filter.add(face_expression)

        result = {
            'body_pose': body_pose,
            'global_rot': deepcopy(self.global_rot),
            'global_trans': deepcopy(self.global_trans),
            'input_kinect_joints': kinect_joints,
            'left_hand_pose': left_hand_pose,
            'right_hand_pose': right_hand_pose,
        }
        if is_face_pose:
            result.update({
                'jaw_pose': jaw_pose,
                'face_expression': face_expression
            })

        return result


import torch

from smplx_kinect.common.concater import universe_len

from .model import PosePredictor, RNNPosePredictor


def get_net(args, ndc_train_sample, device):
    net_input_ndc, net_target_ndc, meta = ndc_train_sample

    # if args['ief']:
    #     assert init_smplx is not None
    #     net_input = concat_init_smplx(kinect_joints, init_smplx)
    #     in_features = universe_len(net_input)
    # else:
    #     in_features = universe_len(kinect_joints.data, is_seq=True)
    #     print('In features', in_features)

    in_features = universe_len(net_input_ndc.data, is_seq=True)
    print('In features', in_features)

    heads = {
        'body_pose': universe_len(net_target_ndc.data, is_seq=True),
        # 'global_rot': universe_len(target_smplx.name2data('global_rot')),
        # 'global_trans': universe_len(target_smplx.name2data('global_trans'))
    }

    print('Heads: ', heads)

    if args['seq_len'] == 1:
        print('MLP', args['mlp_layers'])
        net = PosePredictor(
            hiddens=[in_features] + args['mlp_layers'],
            heads=heads
        )
    else:
        print('RNN', args['mlp_layers'])
        net = RNNPosePredictor(
            input_size=in_features,
            heads=heads,
            mlp_layers=args['mlp_layers']
        )

    if args['load_params_fp'] is not None:
        loaded = torch.load(args['load_params_fp'], map_location=device)
        if args['load_param_exclude'] is not None:
            for k in args['load_param_exclude']:
                del loaded[k]
        net.load_state_dict(loaded, strict=False)
        print('weights loaded')
    net.to(device)

    return net
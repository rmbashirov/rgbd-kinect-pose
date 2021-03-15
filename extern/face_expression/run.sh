python train.py log.experiment_name=debug
python train.py log.experiment_name=debug data.train.n_iters_per_epoch=1 data.val.n_iters_per_epoch=1

python train.py log.experiment_name=siamese_dropout model=siamese_dropout
python train.py log.experiment_name=siamese_small model=siamese_small criterion.jaw_pose.weight=1.0

# overfit
python train.py \
    log.experiment_name=overfit \
    model=siamese_mediapipe_2d \
    log.log_freq_image_batch.train=20 \
    data.train.dataloader.args.batch_size=4 \
    data.train.dataloader.args.shuffle=false \
    data.train.dataset.args.sample_range=[0,4] \
    data.val.dataloader.args.batch_size=4 \
    data.val.dataset.args.sample_range=[0,4]

# after bbox filter
python train.py log.experiment_name=siamese_small_bbox_filter model=siamese_small criterion.jaw_pose.weight=1.0
python train.py log.experiment_name=siamese_dropout_bbox_filter model=siamese_dropout criterion.jaw_pose.weight=1.0
python train.py log.experiment_name=siamese_dropout_bbox_filter model=siamese_dropout criterion.jaw_pose.weight=1.0

# jaw pose weight
python train.py log.experiment_name=siamese_jaw_pose_weight-10.0 model=siamese_dropout criterion.jaw_pose.weight=10.0
python train.py log.experiment_name=siamese_jaw_pose_weight-20.0 model=siamese_dropout criterion.jaw_pose.weight=20.0
python train.py log.experiment_name=siamese_jaw_pose_weight-10.0_lr-0.001 model=siamese_dropout criterion.jaw_pose.weight=10.0 optimizer.predictor.args.lr=0.001

# normalize_area: False
python train.py \
    log.experiment_name=siamese_normalize_area-False_jaw_pose_weight-10.0 \
    criterion.jaw_pose.weight=10.0 \
    data.train.dataset.args.normalize_area=false \
    data.val.dataset.args.normalize_area=false

# keypoint_l2 loss
python train.py \
    log.experiment_name=siamese_keypoint_l2_normalize_area-False \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5 \
    data.train.dataset.args.normalize_area=false \
    data.val.dataset.args.normalize_area=false

# siamese+keypoint_l2_loss+normalize-image_shape
python train.py \
    log.experiment_name="siamese+keypoint_l2_loss+normalize-image_shape" \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5 \
    preprocessing.keypoints_2d_normalization="image_shape"

# expression_weight-10
python train.py \
    log.experiment_name="expression_weight-10" \
    criterion.expression.weight=10.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5 \
    preprocessing.keypoints_2d_normalization="image_shape"

# use_beta-false
python train.py \
    log.experiment_name="use_beta-false" \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5 \
    preprocessing.keypoints_2d_normalization="image_shape"


### mediapipe_normalization

# siamese+mediapipe_normalization
python train.py \
    log.experiment_name="siamese+mediapipe_normalization" \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5

# siamese+mediapipe_normalization+expression_weight-10
python train.py \
    log.experiment_name="siamese+mediapipe_normalization+expression_weight-10" \
    criterion.expression.weight=10.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5

# siamese+mediapipe_normalization+use_beta-false
python train.py \
    log.experiment_name="siamese+mediapipe_normalization+use_beta-false" \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5

# siamese+mediapipe_normalization+no_keypoint_l2_loss
python train.py \
    log.experiment_name="siamese+mediapipe_normalization+no_keypoint_l2_loss" \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.0

# siamese+keypoints_3d
python train.py \
    log.experiment_name="siamese+keypoints_3d" \
    model.predictor.args.use_keypoints_3d=true \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5

# siamese+keypoints_3d+tf-1.15
python train.py \
    log.experiment_name="siamese+keypoints_3d+tf-1.15" \
    model.predictor.args.use_keypoints_3d=true \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5

# siamese+keypoints_3d+expression_weight-10
python train.py \
    log.experiment_name="siamese+keypoints_3d+expression_weight-10" \
    model.predictor.args.use_keypoints_3d=true \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=10.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_l2.weight=0.5

# siamese+keypoints_3d_loss_weigth-1000
python train.py \
    log.experiment_name="siamese+keypoints_3d_loss" \
    model.predictor.args.use_keypoints_3d=true \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_2d_l2.weight=0.0 \
    criterion.keypoint_3d_l2.weight=1000.0

# siamese+keypoints_3d_loss+expression_loss
python train.py \
    log.experiment_name="siamese+keypoints_3d_loss+expression_loss" \
    model.predictor.args.use_keypoints_3d=true \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=0.0 \
    criterion.keypoint_2d_l2.weight=0.0 \
    criterion.keypoint_3d_l2.weight=1000.0

# siamese+mouth
python train.py \
    log.experiment_name="siamese+mouth" \
    runner=runner_mouth \
    criterion=criterion_mouth \
    model.predictor.args.use_keypoints_3d=true \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_2d_l2.weight=0.0 \
    criterion.keypoint_3d_l2.weight=0.0 \
    criterion.keypoint_3d_mouth_l2.weight=3000.0

# siamese+keypoints_3d
python train.py \
    log.experiment_name="siamese+keypoints_3d" \
    model.predictor.args.use_keypoints_3d=true \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=10.0 \
    criterion.jaw_pose.weight=10.0 \
    criterion.keypoint_2d_l2.weight=0.5 \
    criterion.keypoint_3d_l2.weight=0.0

# siamese+mouth+keypoints_3d_loss+expression_loss
python train.py \
    log.experiment_name="siamese+mouth+keypoints_3d_loss+expression_loss" \
    runner=runner_mouth \
    criterion=criterion_mouth \
    model.predictor.args.use_keypoints_3d=true \
    model.predictor.args.use_beta=false \
    criterion.expression.weight=1.0 \
    criterion.jaw_pose.weight=0.0 \
    criterion.keypoint_2d_l2.weight=0.0 \
    criterion.keypoint_3d_l2.weight=1000.0 \
    criterion.keypoint_3d_mouth_l2.weight=5000.0
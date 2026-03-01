Upload a dataset
`hf upload dopaul/pcb_placement_100x_1st_item /home/dominique/.cache/huggingface/lerobot/dopaul/pcb_placement_100x_1st_item . --repo-type dataset`

Download a datset
`hf download dopaul/DATASET_NAME --repo-type dataset --local-dir /home/dominique/.cache/huggingface/lerobot/dopaul/DATASET_NAME`

Upload a model
`hf upload dopaul/pcb_placement_v1_act_001000 /teamspace/jobs/act-50-pcb-placing-samples/artifacts/ledream/outputs/train/pcb_placement_v1_act_baseline/checkpoints/001000/pretrained_model`

Maybe do this first 
`hf auth login`

# Training Models

## ACT

Train (baseline)

```
HF_USER=dopaul
DATASET=pcb_placement_100x_1st_item
JOB_NAME=pcb_placement_1st_item_act_48acl
BASE_REPO=pcb_placement_1st_item_act_48acl

lerobot-train \
  --dataset.repo_id=${HF_USER}/${DATASET} \
  --policy.type=act \
  --output_dir=outputs/train/${JOB_NAME} \
  --job_name=${JOB_NAME} \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.chunk_size=48 \
  --policy.n_action_steps=30 \
  --policy.use_vae=true \
  --policy.kl_weight=10 \
  --batch_size=16 \
  --policy.optimizer_lr=3e-5 \
  --policy.optimizer_lr_backbone=3e-5 \
  --steps=10000 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=2000 \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=${HF_USER}/act_policy

# Upload selected checkpoints to separate model repos.
for CKPT in 002000 004000 006000 008000 010000; do
  hf upload ${HF_USER}/${BASE_REPO}-${CKPT} \
    outputs/train/${JOB_NAME}/checkpoints/${CKPT}/pretrained_model \
    . \
    --repo-type model
done
```

### Run ACT inference/eval (real robot)

```
rm -rf /home/dominique/.cache/huggingface/lerobot/dopaul/eval_pcb_placement_v1_act_eval_002000
HF_USER=dopaul
POLICY_REPO=${HF_USER}/pcb_placement_v1_act_002000
LEFT_FOLLOWER_PORT=/dev/serial/by-path/platform-a80aa10000.usb-usb-0:4.2.1.4:1.0
RIGHT_FOLLOWER_PORT=/dev/serial/by-path/platform-a80aa10000.usb-usb-0:4.2.1.1:1.0
CAM_WIDTH=640
CAM_HEIGHT=480
CAM_FPS=30
CAM_TOP=/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:3.1:1.0-video-index0
CAM_LEFT=/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.1.2:1.0-video-index0
CAM_RIGHT=/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.1.3:1.0-video-index0

uv run lerobot-record \
  --robot.type=bi_dk1_follower \
  --robot.left_arm_port=${LEFT_FOLLOWER_PORT} \
  --robot.right_arm_port=${RIGHT_FOLLOWER_PORT} \
  --robot.id=my_robot_id \
  --policy.path=${POLICY_REPO} \
  --dataset.repo_id=${HF_USER}/eval_pcb_placement_v1_act_eval_002000 \
  --dataset.single_task="Take a PCB from the box and place it in the testbed" \
  --dataset.num_episodes=10 \
  --robot.cameras='{top: {type: opencv, index_or_path: "'$CAM_TOP'", width: '$CAM_WIDTH', height: '$CAM_HEIGHT', fps: '$CAM_FPS', backend: 200, fourcc: MJPG}, left_wrist: {type: opencv, index_or_path: "'$CAM_LEFT'", width: '$CAM_WIDTH', height: '$CAM_HEIGHT', fps: '$CAM_FPS', backend: 200, fourcc: MJPG}, right_wrist: {type: opencv, index_or_path: "'$CAM_RIGHT'", width: '$CAM_WIDTH', height: '$CAM_HEIGHT', fps: '$CAM_FPS', backend: 200, fourcc: MJPG}}' \
  --dataset.episode_time_s=25 \
	--dataset.reset_time_s=0 \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=1 \
  --dataset.push_to_hub=false \
  --dataset.vcodec=auto \
  --play_sounds=false \
  --display_data=false \
  --robot.joint_velocity_scaling=0.5
```

# Train Diffusion Policy (DP)

```
HF_USER=dopaul
DATASET=pcb_placement_100x_1st_item
POLICY_REPO=${HF_USER}/pcb_placement_v1_diffusion_policy
JOB_NAME=pcb_placement_100x_1st_item_diffusion
BASE_REPO=pcb_placement_100x_1st_item_diffusion

lerobot-train \
  --dataset.repo_id=dopaul/pcb_placement_100x_1st_item \
  --policy.type=diffusion \
  --output_dir=outputs/train/pcb_placement_100x_1st_item_diffusion \
  --job_name=pcb_placement_100x_1st_item_diffusion \
  --policy.repo_id=dopaul/pcb_placement_v1_diffusion_policy \
  --policy.device=cuda \
  --policy.use_amp=true \
  --batch_size=64 \
  --num_workers=0 \
  --steps=5000 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --wandb.enable=true

# Upload selected checkpoints to separate model repos.
for CKPT in 001000 002000 003000 004000 005000; do
  hf upload ${HF_USER}/${BASE_REPO}-${CKPT} \
    outputs/train/${JOB_NAME}/checkpoints/${CKPT}/pretrained_model \
    . \
    --repo-type model
done

# Train pi0.5 (pi05)

The guidelines provided by PI is between 1 to 20 hours of finetuning data for task adaptation.

```
# One-time setup (if needed)
# pip install -e ".[pi]"
# hf auth login
# hf auth whoami
#
# IMPORTANT: request and approve access to gated model:
# https://huggingface.co/google/paligemma-3b-pt-224
# (same HF account used by hf auth login)

HF_USER=dopaul
DATASET=pcb_placement_v1
JOB_NAME=pcb_placement_v1_pi05
POLICY_REPO=${HF_USER}/pcb_placement_v1_pi05_policy

# Required for default QUANTILES normalization (q01/q99)
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
  --repo-id=${HF_USER}/${DATASET}

lerobot-train \
  --dataset.repo_id=${HF_USER}/${DATASET} \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --output_dir=outputs/train/${JOB_NAME} \
  --job_name=${JOB_NAME} \
  --policy.repo_id=${POLICY_REPO} \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=3000 \
  --save_checkpoint=true \
  --save_freq=500 \
  --wandb.enable=true

# Upload selected checkpoints to separate model repos.
BASE_REPO=pcb_placement_v1_pi05
for CKPT in 000500 001000 001500 002000 002500 003000; do
  huggingface-cli upload ${HF_USER}/${BASE_REPO}-${CKPT} \
    outputs/train/${JOB_NAME}/checkpoints/${CKPT}/pretrained_model \
    . \
    --repo-type model
done
```

Run pi0.5 inference/eval (real robot)

```
HF_USER=dopaul
POLICY_REPO=${HF_USER}/pcb_placement_v1_pi05_policy

lerobot-record \
  --robot.type=your_robot_type \
  --robot.port=/dev/ttyACM1 \
  --robot.id=my_robot_id \
  --policy.path=${POLICY_REPO} \
  --dataset.repo_id=${HF_USER}/pcb_placement_v1_pi05_eval \
  --dataset.single_task="pcb placement" \
  --dataset.num_episodes=10
```
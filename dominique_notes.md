Upload a dataset
`huggingface-cli upload dopaul/DATASET_NAME /home/dominique/.cache/huggingface/lerobot/dopaul/DATASET_NAME . --repo-type dataset`

Download a datset
`hf download dopaul/DATASET_NAME --repo-type dataset --local-dir /home/dominique/.cache/huggingface/lerobot/dopaul/DATASET_NAME`

Maybe do this first 
`hf auth login`

# Training Models

Train ACT (baseline)

```
HF_USER=dopaul
DATASET=pcb_placement_v1
JOB_NAME=pcb_placement_v1_act_baseline

lerobot-train \
  --dataset.repo_id=${HF_USER}/${DATASET} \
  --policy.type=act \
  --output_dir=outputs/train/${JOB_NAME} \
  --job_name=${JOB_NAME} \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.chunk_size=30 \
  --policy.n_action_steps=30 \
  --policy.use_vae=true \
  --policy.kl_weight=10 \
  --batch_size=96 \
  --policy.optimizer_lr=2.4e-4 \
  --policy.optimizer_lr_backbone=2.4e-4 \
  --steps=5000 \
  --log_freq=100 \
  --save_checkpoint=true \
  --save_freq=1000 \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id=${HF_USER}/act_policy

# Upload selected checkpoints to separate model repos.
BASE_REPO=pcb_placement_v1_act_baseline
for CKPT in 001000 002000 003000 004000 005000; do
  huggingface-cli upload ${HF_USER}/${BASE_REPO}-${CKPT} \
    outputs/train/${JOB_NAME}/checkpoints/${CKPT}/pretrained_model \
    . \
    --repo-type model
done
```

# Train pi0.5 (pi05)

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
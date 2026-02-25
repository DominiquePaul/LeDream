# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: tests teleop-arms record-arms

PYTHON_PATH := $(shell which python)

# If uv is installed and a virtual environment exists, use it
UV_CHECK := $(shell command -v uv)
ifneq ($(UV_CHECK),)
	PYTHON_PATH := .venv/bin/python
endif

export PATH := $(dir $(PYTHON_PATH)):$(PATH)

DEVICE ?= cpu

# --- Added by DP START ---

# Fixed default ports for bimanual arm setup (Jetson/Linux).
# Prefer stable symlinks over volatile ttyACM* names.
LEFT_LEADER_PORT ?= /dev/serial/by-id/usb-1a86_USB_Single_Serial_5A46081965-if00
RIGHT_LEADER_PORT ?= /dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0181138-if00
LEFT_FOLLOWER_PORT ?= /dev/serial/by-path/platform-a80aa10000.usb-usb-0:4.2.1.4:1.0
RIGHT_FOLLOWER_PORT ?= /dev/serial/by-path/platform-a80aa10000.usb-usb-0:4.2.1.1:1.0

# Recording defaults.
DATASET_REPO_ID ?= local/pcb_basic
DATASET_TASK ?= Place PCB into testing device, wait, and place into right box.
DATASET_NUM_EPISODES ?= 50
DATASET_EPISODE_TIME_S ?= 1200
DATASET_RESET_TIME_S ?= 0.0
FPS ?= 30
DISPLAY_DATA ?= false
JOINT_VELOCITY_SCALING ?= 1.0

# --- Added by DP END ---


build-user:
	docker build -f docker/Dockerfile.user -t lerobot-user .

build-internal:
	docker build -f docker/Dockerfile.internal -t lerobot-internal .

test-end-to-end:
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train-resume
	${MAKE} DEVICE=$(DEVICE) test-act-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-train
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-train
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-train
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-eval

test-act-ete-train:
	lerobot-train \
		--policy.type=act \
		--policy.dim_model=64 \
		--policy.n_action_steps=20 \
		--policy.chunk_size=20 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=aloha \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=4 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_freq=2 \
		--save_checkpoint=true \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/act/

test-act-ete-train-resume:
	lerobot-train \
		--config_path=tests/outputs/act/checkpoints/000002/pretrained_model/train_config.json \
		--resume=true

test-act-ete-eval:
	lerobot-eval \
		--policy.path=tests/outputs/act/checkpoints/000004/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=aloha \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

test-diffusion-ete-train:
	lerobot-train \
		--policy.type=diffusion \
		--policy.down_dims='[64,128,256]' \
		--policy.diffusion_step_embed_dim=32 \
		--policy.num_inference_steps=10 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=pusht \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/pusht \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=2 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_checkpoint=true \
		--save_freq=2 \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/diffusion/

test-diffusion-ete-eval:
	lerobot-eval \
		--policy.path=tests/outputs/diffusion/checkpoints/000002/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=pusht \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

test-tdmpc-ete-train:
	lerobot-train \
		--policy.type=tdmpc \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=pusht \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/pusht_image \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=2 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_checkpoint=true \
		--save_freq=2 \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/tdmpc/

test-tdmpc-ete-eval:
	lerobot-eval \
		--policy.path=tests/outputs/tdmpc/checkpoints/000002/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=pusht \
		--env.episode_length=5 \
		--env.observation_height=96 \
        --env.observation_width=96 \
		--eval.n_episodes=1 \
		--eval.batch_size=1


test-smolvla-ete-train:
	lerobot-train \
		--policy.type=smolvla \
		--policy.n_action_steps=20 \
		--policy.chunk_size=20 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=aloha \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=4 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_freq=2 \
		--save_checkpoint=true \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/smolvla/

test-smolvla-ete-eval:
	lerobot-eval \
		--policy.path=tests/outputs/smolvla/checkpoints/000004/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=aloha \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

teleop-arms:
	lerobot-teleoperate \
		--robot.type=bi_so_follower \
		--robot.left_arm_config.port=$(LEFT_FOLLOWER_PORT) \
		--robot.right_arm_config.port=$(RIGHT_FOLLOWER_PORT) \
		--teleop.type=bi_so_leader \
		--teleop.left_arm_config.port=$(LEFT_LEADER_PORT) \
		--teleop.right_arm_config.port=$(RIGHT_LEADER_PORT) \
		--robot.joint_velocity_scaling=$(JOINT_VELOCITY_SCALING) \
		--fps=$(FPS) \
		--display_data=$(DISPLAY_DATA)

record-arms:
	lerobot-record \
		--robot.type=bi_so_follower \
		--robot.left_arm_config.port=$(LEFT_FOLLOWER_PORT) \
		--robot.right_arm_config.port=$(RIGHT_FOLLOWER_PORT) \
		--teleop.type=bi_so_leader \
		--teleop.left_arm_config.port=$(LEFT_LEADER_PORT) \
		--teleop.right_arm_config.port=$(RIGHT_LEADER_PORT) \
		--robot.joint_velocity_scaling=$(JOINT_VELOCITY_SCALING) \
		--fps=$(FPS) \
		--display_data=$(DISPLAY_DATA) \
		--dataset.repo_id=$(DATASET_REPO_ID) \
		--dataset.single_task="$(DATASET_TASK)" \
		--dataset.num_episodes=$(DATASET_NUM_EPISODES) \
		--dataset.episode_time_s=$(DATASET_EPISODE_TIME_S) \
		--dataset.reset_time_s=$(DATASET_RESET_TIME_S) \
		--dataset.push_to_hub=false

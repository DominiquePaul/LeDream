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

.PHONY: tests teleop-arms record-arms biteleop biterecord birest snapshot cam-tuner

PYTHON_PATH := $(shell which python)

# Prefer the project virtualenv whenever it exists.
ifneq ("$(wildcard .venv/bin/python)","")
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
DATASET_REPO_ID ?= dopaul/pcb_placement_100x_1st_item
DATASET_TASK ?= Take a PCB from the box and place it in the testbed
DATASET_NUM_EPISODES ?= 1
DATASET_EPISODE_TIME_S ?= 35
DATASET_RESET_TIME_S ?= 7.0
FPS ?= 30
DISPLAY_DATA ?= false
JOINT_VELOCITY_SCALING ?= 1.0
JETSON_MAX_PERF ?= true
resume ?= false
REST_MOVE_DURATION_S ?= 3.0
REST_STEPS ?= 90
# Joint order: joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,gripper
LEFT_REST_POSE ?= 0,0,0,0,0,0,0
RIGHT_REST_POSE ?= 0,0,0,0,0,0,0

# Camera paths (stable by-path symlinks).
CAM_TOP ?= /dev/v4l/by-path/platform-a80aa10000.usb-usb-0:3.1:1.0-video-index0
CAM_LEFT ?= /dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.1.2:1.0-video-index0
CAM_RIGHT ?= /dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.1.3:1.0-video-index0
CAM_WIDTH ?= 640
CAM_HEIGHT ?= 480
CAM_FPS ?= 30
SNAPSHOT_ARGS ?=

# --- Added by DP END ---

snapshot:
	python scripts/snapshot.py \
		--cameras "top=$(CAM_TOP),left_wrist=$(CAM_LEFT),right_wrist=$(CAM_RIGHT)" \
		--width $(CAM_WIDTH) \
		--height $(CAM_HEIGHT) \
		$(SNAPSHOT_ARGS)

cam-tuner:
	python scripts/cam_tuner.py \
		--cameras "top=$(CAM_TOP),left_wrist=$(CAM_LEFT),right_wrist=$(CAM_RIGHT)"

build-user:
	docker build -f docker/Dockerfile.user -t lerobot-user .

build-internal:
	docker build -f docker/Dockerfile.internal -t lerobot-internal .

test-end-to-end:
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train-resume
	${MAKE} DEVICE=$(DEVICE) test-act-ete-evaly
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
		--dataset.fps=$(FPS) \
		--display_data=$(DISPLAY_DATA) \
		--dataset.repo_id=$(DATASET_REPO_ID) \
		--dataset.single_task="$(DATASET_TASK)" \
		--dataset.num_episodes=$(DATASET_NUM_EPISODES) \
		--dataset.episode_time_s=$(DATASET_EPISODE_TIME_S) \
		--dataset.reset_time_s=$(DATASET_RESET_TIME_S) \
		--dataset.push_to_hub=false

biteleop:
	lerobot-teleoperate \
		--robot.type=bi_dk1_follower \
		--teleop.type=bi_dk1_leader \
		--teleop.left_arm_port=$(LEFT_LEADER_PORT) \
		--robot.left_arm_port=$(LEFT_FOLLOWER_PORT) \
		--teleop.right_arm_port=$(RIGHT_LEADER_PORT) \
		--robot.right_arm_port=$(RIGHT_FOLLOWER_PORT) \
		--robot.joint_velocity_scaling=$(JOINT_VELOCITY_SCALING) \
		--fps=$(FPS) \
		--display_data=$(DISPLAY_DATA)

birest:
	@LEFT_REST_POSE="$(LEFT_REST_POSE)" \
	RIGHT_REST_POSE="$(RIGHT_REST_POSE)" \
	REST_MOVE_DURATION_S="$(REST_MOVE_DURATION_S)" \
	REST_STEPS="$(REST_STEPS)" \
	LEFT_FOLLOWER_PORT="$(LEFT_FOLLOWER_PORT)" \
	RIGHT_FOLLOWER_PORT="$(RIGHT_FOLLOWER_PORT)" \
	JOINT_VELOCITY_SCALING="$(JOINT_VELOCITY_SCALING)" \
	$(PYTHON_PATH) scripts/birest.py

birecord:
	@if [ "$(JETSON_MAX_PERF)" = "true" ]; then \
		sudo -n nvpmodel -m 0 >/dev/null 2>&1 || echo "Skipping nvpmodel: sudo rights required"; \
		sudo -n jetson_clocks >/dev/null 2>&1 || echo "Skipping jetson_clocks: sudo rights required"; \
	fi
	lerobot-record \
		--robot.type=bi_dk1_follower \
		--teleop.type=bi_dk1_leader \
		--teleop.left_arm_port=$(LEFT_LEADER_PORT) \
		--robot.left_arm_port=$(LEFT_FOLLOWER_PORT) \
		--teleop.right_arm_port=$(RIGHT_LEADER_PORT) \
		--robot.right_arm_port=$(RIGHT_FOLLOWER_PORT) \
		--robot.joint_velocity_scaling=$(JOINT_VELOCITY_SCALING) \
		--robot.cameras='{top: {type: opencv, index_or_path: "$(CAM_TOP)", width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS), backend: 200, fourcc: MJPG}, left_wrist: {type: opencv, index_or_path: "$(CAM_LEFT)", width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS), backend: 200, fourcc: MJPG}, right_wrist: {type: opencv, index_or_path: "$(CAM_RIGHT)", width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS), backend: 200, fourcc: MJPG}}' \
		--dataset.fps=$(FPS) \
		--display_data=$(DISPLAY_DATA) \
		--dataset.repo_id=$(DATASET_REPO_ID) \
		--dataset.single_task="$(DATASET_TASK)" \
		--dataset.num_episodes=$(DATASET_NUM_EPISODES) \
		--dataset.episode_time_s=$(DATASET_EPISODE_TIME_S) \
		--dataset.reset_time_s=$(DATASET_RESET_TIME_S) \
		--dataset.push_to_hub=false \
		--dataset.streaming_encoding=true \
		--dataset.encoder_threads=1 \
		--dataset.vcodec=auto \
		--play_sounds=false \
		--resume=$(resume)

birecord3o:
	lerobot-record \
		--robot.type=bi_dk1_follower \
		--teleop.type=bi_dk1_leader \
		--teleop.left_arm_port=$(LEFT_LEADER_PORT) \
		--robot.left_arm_port=$(LEFT_FOLLOWER_PORT) \
		--teleop.right_arm_port=$(RIGHT_LEADER_PORT) \
		--robot.right_arm_port=$(RIGHT_FOLLOWER_PORT) \
		--robot.joint_velocity_scaling=$(JOINT_VELOCITY_SCALING) \
		--robot.cameras='{top: {type: opencv, index_or_path: "$(CAM_TOP)", width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS)}}' \
		--dataset.fps=$(FPS) \
		--display_data=$(DISPLAY_DATA) \
		--dataset.repo_id=$(DATASET_REPO_ID) \
		--dataset.single_task="$(DATASET_TASK)" \
		--dataset.num_episodes=$(DATASET_NUM_EPISODES) \
		--dataset.episode_time_s=$(DATASET_EPISODE_TIME_S) \
		--dataset.reset_time_s=$(DATASET_RESET_TIME_S) \
		--dataset.push_to_hub=true \
		--dataset.streaming_encoding=true \
		--dataset.encoder_threads=4 \
		--play_sounds=false




birecord2o:
	lerobot-record \
		--robot.type=bi_dk1_follower \
		--teleop.type=bi_dk1_leader \
		--teleop.left_arm_port=$(LEFT_LEADER_PORT) \
		--robot.left_arm_port=$(LEFT_FOLLOWER_PORT) \
		--teleop.right_arm_port=$(RIGHT_LEADER_PORT) \
		--robot.right_arm_port=$(RIGHT_FOLLOWER_PORT) \
		--robot.joint_velocity_scaling=$(JOINT_VELOCITY_SCALING) \
		--robot.cameras='{right_wrist: {type: opencv, index_or_path: "$(CAM_RIGHT)", width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS), backend: 200, fourcc: MJPG}}' \
		--dataset.fps=$(FPS) \
		--display_data=$(DISPLAY_DATA) \
		--dataset.repo_id=$(DATASET_REPO_ID) \
		--dataset.single_task="$(DATASET_TASK)" \
		--dataset.num_episodes=$(DATASET_NUM_EPISODES) \
		--dataset.episode_time_s=$(DATASET_EPISODE_TIME_S) \
		--dataset.reset_time_s=$(DATASET_RESET_TIME_S) \
		--dataset.push_to_hub=true \
		--dataset.streaming_encoding=true \
		--dataset.encoder_threads=2 \
		--play_sounds=false

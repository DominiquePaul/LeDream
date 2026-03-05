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

.PHONY: tests teleop-arms record-arms biteleop biterecord birest snapshot \
       list-jobs list-checkpoints upload-latest \
       list-configs train train-cmd eval

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

# ── HF checkpoint upload helpers ──────────────────────────────────────────────
JOBS_ROOT ?= /teamspace/jobs
HF_USER   ?= dopaul

# List all jobs and their latest checkpoint.
#   make list-jobs
list-jobs:
	@printf "%-30s %-50s %s\n" "JOB" "TRAIN_NAME" "LATEST_CKPT"; \
	printf "%-30s %-50s %s\n" "---" "----------" "-----------"; \
	for job in $(JOBS_ROOT)/*/; do \
		name=$$(basename "$$job"); \
		ckpt_dir=$$(ls -d "$$job"artifacts/ledream/outputs/train/*/checkpoints 2>/dev/null | head -1); \
		if [ -n "$$ckpt_dir" ]; then \
			latest=$$(ls "$$ckpt_dir" 2>/dev/null | grep -v last | sort -n | tail -1); \
			train_name=$$(basename "$$(dirname "$$ckpt_dir")"); \
			printf "%-30s %-50s %s\n" "$$name" "$$train_name" "$$latest"; \
		else \
			printf "%-30s %-50s %s\n" "$$name" "(no checkpoints)" "-"; \
		fi; \
	done

# List checkpoints for a specific job.
#   make list-checkpoints JOB=100x-pcb-act-b16
list-checkpoints:
	@if [ -z "$(JOB)" ]; then echo "ERROR: set JOB=<job-name>  (run 'make list-jobs' to see options)"; exit 1; fi
	@ckpt_dir=$$(ls -d $(JOBS_ROOT)/$(JOB)/artifacts/ledream/outputs/train/*/checkpoints 2>/dev/null | head -1); \
	if [ -z "$$ckpt_dir" ]; then echo "No checkpoints found for JOB=$(JOB)"; exit 1; fi; \
	echo "Checkpoints for $(JOB) ($$ckpt_dir):"; \
	ls "$$ckpt_dir"

# Upload the latest (or a specific) checkpoint to Hugging Face.
#   make upload-latest JOB=100x-pcb-act-b16                   # latest ckpt, auto repo name
#   make upload-latest JOB=100x-pcb-act-b16 CKPT=004000       # specific ckpt
#   make upload-latest JOB=100x-pcb-act-b16 REPO=my_custom_repo  # custom repo name
upload-latest:
	@if [ -z "$(JOB)" ]; then echo "ERROR: set JOB=<job-name>  (run 'make list-jobs' to see options)"; exit 1; fi
	@ckpt_dir=$$(ls -d $(JOBS_ROOT)/$(JOB)/artifacts/ledream/outputs/train/*/checkpoints 2>/dev/null | head -1); \
	if [ -z "$$ckpt_dir" ]; then echo "No checkpoints found for JOB=$(JOB)"; exit 1; fi; \
	train_name=$$(basename "$$(dirname "$$ckpt_dir")"); \
	if [ -n "$(CKPT)" ]; then \
		ckpt="$(CKPT)"; \
	else \
		ckpt=$$(ls "$$ckpt_dir" | grep -v last | sort -n | tail -1); \
	fi; \
	model_path="$$ckpt_dir/$$ckpt/pretrained_model"; \
	if [ ! -d "$$model_path" ]; then echo "ERROR: $$model_path does not exist"; exit 1; fi; \
	repo_name="$${train_name}-$${ckpt}"; \
	if [ -n "$(REPO)" ]; then repo_name="$(REPO)"; fi; \
	echo ""; \
	echo "  Job:        $(JOB)"; \
	echo "  Train:      $$train_name"; \
	echo "  Checkpoint:  $$ckpt"; \
	echo "  Model path:  $$model_path"; \
	echo "  HF repo:    $(HF_USER)/$$repo_name"; \
	echo ""; \
	hf upload $(HF_USER)/$$repo_name "$$model_path" . --repo-type model

# ── Training config management ────────────────────────────────────────────────
CONFIGS_DIR ?= configs
OVERRIDES   ?=

# List available training configs.
#   make list-configs
list-configs:
	@printf "%-25s %-12s %s\n" "CONFIG" "POLICY" "DATASET"; \
	printf "%-25s %-12s %s\n" "------" "------" "-------"; \
	for f in $(CONFIGS_DIR)/*.json; do \
		name=$$(basename "$$f" .json); \
		policy=$$($(PYTHON_PATH) -c "import json; print(json.load(open('$$f')).get('policy',{}).get('type','?'))"); \
		dataset=$$($(PYTHON_PATH) -c "import json; print(json.load(open('$$f')).get('dataset',{}).get('repo_id','?'))"); \
		printf "%-25s %-12s %s\n" "$$name" "$$policy" "$$dataset"; \
	done

# Print the full training command (for pasting into Lightning AI job UI).
#   make train-cmd CONFIG=pcb100x_act
#   make train-cmd CONFIG=pcb100x_act OVERRIDES="--steps=50000 --batch_size=32"
train-cmd:
	@if [ -z "$(CONFIG)" ]; then echo "ERROR: set CONFIG=<name>  (run 'make list-configs')"; exit 1; fi
	@if [ ! -f "$(CONFIGS_DIR)/$(CONFIG).json" ]; then echo "ERROR: $(CONFIGS_DIR)/$(CONFIG).json not found"; exit 1; fi
	@echo "lerobot-train \\"
	@echo "  --config_path=$(CONFIGS_DIR)/$(CONFIG).json \\"
	@echo "  --output_dir=outputs/train/$(CONFIG) \\"
	@echo "  --job_name=$(CONFIG) \\"
	@echo "  $(OVERRIDES)"

# Run training locally.
#   make train CONFIG=pcb100x_act
#   make train CONFIG=pcb100x_act OVERRIDES="--steps=50000 --batch_size=32"
train:
	@if [ -z "$(CONFIG)" ]; then echo "ERROR: set CONFIG=<name>  (run 'make list-configs')"; exit 1; fi
	@if [ ! -f "$(CONFIGS_DIR)/$(CONFIG).json" ]; then echo "ERROR: $(CONFIGS_DIR)/$(CONFIG).json not found"; exit 1; fi
	lerobot-train \
		--config_path=$(CONFIGS_DIR)/$(CONFIG).json \
		--output_dir=outputs/train/$(CONFIG) \
		--job_name=$(CONFIG) \
		$(OVERRIDES)

# ── Quick eval on bimanual DK1 robot ─────────────────────────────────────────
EPHEMERAL      ?= false
EVAL_EPISODES  ?= 10
EVAL_TIME_S    ?= 25
EVAL_VELOCITY  ?= 0.5

# Run a trained model on the real robot for quick evaluation.
#   make eval MODEL=dopaul/pcb100x_act-010000
#   make eval MODEL=dopaul/pcb100x_act-010000 EPHEMERAL=true
#   make eval MODEL=dopaul/pcb100x_act-010000 EVAL_EPISODES=5 EVAL_TIME_S=20
eval:
	@if [ -z "$(MODEL)" ]; then echo "ERROR: set MODEL=<hf_repo_id>  (e.g. dopaul/pcb100x_act-010000)"; exit 1; fi
	@model_short=$$(echo "$(MODEL)" | sed 's|.*/||'); \
	eval_dataset="$(HF_USER)/eval_$${model_short}"; \
	local_dir="$${HOME}/.cache/huggingface/lerobot/$${eval_dataset}"; \
	echo ""; \
	echo "  Model:       $(MODEL)"; \
	echo "  Eval dataset: $${eval_dataset}"; \
	echo "  Episodes:    $(EVAL_EPISODES)"; \
	echo "  Ephemeral:   $(EPHEMERAL)"; \
	echo ""; \
	if [ "$(EPHEMERAL)" = "true" ] && [ -d "$$local_dir" ]; then \
		echo "Removing previous eval dataset: $$local_dir"; \
		rm -rf "$$local_dir"; \
	fi; \
	lerobot-record \
		--robot.type=bi_dk1_follower \
		--robot.left_arm_port="$(LEFT_FOLLOWER_PORT)" \
		--robot.right_arm_port="$(RIGHT_FOLLOWER_PORT)" \
		--robot.id=my_robot_id \
		--policy.path="$(MODEL)" \
		--dataset.repo_id="$${eval_dataset}" \
		--dataset.single_task="$(DATASET_TASK)" \
		--dataset.num_episodes=$(EVAL_EPISODES) \
		--robot.cameras='{top: {type: opencv, index_or_path: "$(CAM_TOP)", width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS), backend: 200, fourcc: MJPG}, left_wrist: {type: opencv, index_or_path: "$(CAM_LEFT)", width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS), backend: 200, fourcc: MJPG}, right_wrist: {type: opencv, index_or_path: "$(CAM_RIGHT)", width: $(CAM_WIDTH), height: $(CAM_HEIGHT), fps: $(CAM_FPS), backend: 200, fourcc: MJPG}}' \
		--dataset.episode_time_s=$(EVAL_TIME_S) \
		--dataset.reset_time_s=0 \
		--dataset.streaming_encoding=true \
		--dataset.encoder_threads=1 \
		--dataset.push_to_hub=false \
		--dataset.vcodec=auto \
		--play_sounds=false \
		--display_data=false \
		--robot.joint_velocity_scaling=$(EVAL_VELOCITY); \
	rc=$$?; \
	if [ "$(EPHEMERAL)" = "true" ] && [ -d "$$local_dir" ]; then \
		echo "Cleaning up eval dataset: $$local_dir"; \
		rm -rf "$$local_dir"; \
	fi; \
	exit $$rc

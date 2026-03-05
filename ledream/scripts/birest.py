#!/usr/bin/env python3

import os
import time

from lerobot.robots import make_robot_from_config
from lerobot_robot_trlc_dk1.bi_follower import BiDK1FollowerConfig


def parse_pose(env_name: str) -> list[float]:
    values = [float(x.strip()) for x in os.environ[env_name].split(",") if x.strip()]
    if len(values) != 7:
        raise ValueError(f"{env_name} must contain 7 comma-separated values")
    return values


def main() -> None:
    left_pose = parse_pose("LEFT_REST_POSE")
    right_pose = parse_pose("RIGHT_REST_POSE")
    duration_s = float(os.environ["REST_MOVE_DURATION_S"])
    steps = max(1, int(float(os.environ["REST_STEPS"])))
    velocity_scale = float(os.environ["JOINT_VELOCITY_SCALING"])

    keys = [f"joint_{i}.pos" for i in range(1, 7)] + ["gripper.pos"]
    target = {f"left_{k}": v for k, v in zip(keys, left_pose)} | {
        f"right_{k}": v for k, v in zip(keys, right_pose)
    }

    robot_cfg = BiDK1FollowerConfig(
        left_arm_port=os.environ["LEFT_FOLLOWER_PORT"],
        right_arm_port=os.environ["RIGHT_FOLLOWER_PORT"],
        joint_velocity_scaling=velocity_scale,
    )
    robot = make_robot_from_config(robot_cfg)

    print("Connecting bi_dk1_follower...")
    robot.connect()
    try:
        observation = robot.get_observation()
        start = {k: float(observation.get(k, target[k])) for k in target}
        print("Moving to rest pose...")
        for i in range(1, steps + 1):
            alpha = i / steps
            command = {k: start[k] + alpha * (target[k] - start[k]) for k in target}
            robot.send_action(command)
            time.sleep(duration_s / steps)
        print("Rest pose reached.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()

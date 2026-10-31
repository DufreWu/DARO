# ============================================================
# PDQN training script for robot environment mapping
# ============================================================

import os
import csv
import numpy as np

from pdqn_agent import PDQNAgent
from robot_env_tx2 import RobotEnv
from config import config, robot_env


# --------------------------------------------------
# Random external activation (per EPISODE)
# --------------------------------------------------
def sample_app_mask(app_num, p_active=0.6, max_active=3):
    while True:
        mask = (np.random.rand(app_num) < p_active).astype(int)
        active_num = mask.sum()
        if 1 <= active_num <= max_active:
            return mask.tolist()


# --------------------------------------------------
# Training loop
# --------------------------------------------------
def train_pdqn(config):

    # --------------------------------------------------------
    # Environment & Agent
    # --------------------------------------------------------
    env = RobotEnv()

    agent = PDQNAgent(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],            
        param_dim=config["action_parameter_dim"]
    )

    # --------------------------------------------------------
    # Logging setup
    # --------------------------------------------------------
    os.makedirs("./training_results", exist_ok=True)
    reward_log_path = "./training_results/reward.csv"
    state_log_path = "./training_results/env_state.csv"
    reward_writer = None

    with open(reward_log_path, "w", newline="") as f:
        reward_writer = csv.writer(f)
        reward_writer.writerow(["Episode", "Reward"])

        # --------------------------------------------------------
        # Training control
        # --------------------------------------------------------
        episodes = config["episodes"]
        steps_per_episode = config["steps_per_episode"]

        app_mask = sample_app_mask(robot_env["app_num"])
        # app_mask = [1, 1, 0, 1, 0]  # For debugging: fixed active apps

        # initial dummy action to get first state
        action_dim = config["action_parameter_dim"]

        for ep in range(episodes):

            # app_mask = sample_app_mask(robot_env["app_num"])
            state = env.get_sim_current_state(
                action=0,
                action_params=np.zeros(config["action_parameter_dim"]),
                app_mask=app_mask
            )

            ep_reward = 0.0

            for step in range(steps_per_episode):
                action, param = agent.get_action(state)
                next_state = env.get_sim_current_state(action, param, app_mask)

                reward, sys_c, e_term = env.get_reward(
                    next_state,
                    config["beta"],
                    app_mask
                )

                done = (step == steps_per_episode - 1)

                agent.store(state, action, param, reward, next_state, done)
                agent.train()

                state = next_state
                ep_reward += reward

            # -------- logging --------
            ep_avg_reward = ep_reward / steps_per_episode
            print(f"[Episode {ep}] Reward={ep_avg_reward:.3f} | Active={sum(app_mask)}")

            # -------- save --------
            if ep % 100 == 0:
                agent.save("./save_model")

            # ----------------------------------------------------
            # Logging
            # ----------------------------------------------------
            reward_writer.writerow([ep, ep_avg_reward])
            f.flush()

    print("Training completed.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    train_pdqn(config)

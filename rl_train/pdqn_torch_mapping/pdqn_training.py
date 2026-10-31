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
        state_size=config["state_size"],
        action_size=config["action_size"],            # FIXED action space
        action_parameter_size=config["action_parameter_size"]
    )

    # --------------------------------------------------------
    # Logging setup
    # --------------------------------------------------------
    os.makedirs("./training_results", exist_ok=True)
    reward_log_path = "./training_results/reward.csv"
    state_log_path = "./training_results/env_state.csv"

    with open(reward_log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Step", "Episode", "Reward", "sys_reward", "ed_reward", "ActiveApps"]
        )

    with open(state_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = (
            ["Step", "Episode", "Reward", "K"]
            + [f"State{i}" for i in range(config["state_size"])]
            + [f"Mask{i}" for i in range(robot_env["app_num"])]
        )
        writer.writerow(header)

    # --------------------------------------------------------
    # Training control
    # --------------------------------------------------------
    total_steps = config["experiment_time"]
    steps_per_episode = config["steps_per_episode"]

    episode_count = 0
    step_reward_sum = 0.0

    # --------------------------------------------------------
    # Episode initialization
    # --------------------------------------------------------

    # app_mask = sample_app_mask(robot_env["app_num"])
    app_mask = [1, 1, 0, 1, 0]  # For debugging: fixed active apps

    # initial dummy action to get first state
    action_dim = config["action_parameter_size"]
    state = env.get_sim_current_state(
        action=0,
        action_params=np.zeros(action_dim, dtype=np.float32),
        app_mask=app_mask
    )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    for step in range(total_steps):

        # ----------------------------------------------------
        # Agent action
        # ----------------------------------------------------
        action, action_params = agent.get_action(state)

        # ----------------------------------------------------
        # Environment transition
        # ----------------------------------------------------
        next_state = env.get_sim_current_state(
            action,
            action_params,
            app_mask
        )

        reward, sys_c, e_term = env.get_reward(
            next_state,
            config["beta"],
            app_mask
        )

        done = float((step + 1) % steps_per_episode == 0)

        # ----------------------------------------------------
        # Store transition
        # ----------------------------------------------------
        agent.append_sample(
            state,
            action,
            action_params,
            reward,
            next_state,
            done
        )

        agent.train_model()

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------
        with open(reward_log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [step, episode_count, reward, sys_c, e_term, sum(app_mask)]
            )

        with open(state_log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [step, episode_count, reward, sum(app_mask)]
                + next_state.tolist()
                + app_mask
            )

        # ----------------------------------------------------
        # Bookkeeping
        # ----------------------------------------------------
        step_reward_sum += reward
        state = next_state

        # ----------------------------------------------------
        # Episode boundary
        # ----------------------------------------------------
        if done:
            episode_count += 1
            avg_reward = step_reward_sum / steps_per_episode

            print(
                f"[Episode {episode_count}] "
                f"AvgReward = {avg_reward:.3f} | "
                f"ActiveApps = {sum(app_mask)}"
            )

            step_reward_sum = 0.0

            # ---- new episode ----
            app_mask = sample_app_mask(robot_env["app_num"])

            state = env.get_sim_current_state(
                action=0,
                action_params=np.zeros(action_dim, dtype=np.float32),
                app_mask=app_mask
            )

            if episode_count % 10 == 0:
                agent.save_model("./save_model")
                print("Model saved.")

    print("Training completed.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    train_pdqn(config)

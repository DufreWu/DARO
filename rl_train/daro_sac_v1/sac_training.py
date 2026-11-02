from sac_agent import SACAgent, SACCfg, cpu_fre_list, speed_list
from robot_env import RobotEnv
import numpy as np
import csv

state_size = 6
cfg = SACCfg(state_dim=state_size, device="cpu")

# Action ranges (continuous)
action_list = []
for i in range(len(cpu_fre_list)):
    for j in range(len(speed_list)):
        action_list.append((cpu_fre_list[i], speed_list[j]))

print(len(action_list), "actions available")
act_low  = (min(cpu_fre_list), min(speed_list))
act_high = (max(cpu_fre_list), max(speed_list))

agent = SACAgent(cfg, act_low=act_low, act_high=act_high)
env = RobotEnv()

experiment_time = 10000
target_thr = 25

state = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

# Open a CSV file to record rewards
with open('./training_results/reward.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ITERATION', 'REWARD'])  # Write the header

    for t in range(experiment_time):
        # Choose action
        a_cont, (f_cmd, v_cmd) = agent.select_action(state, deterministic=False)
        print(f"Step {t}: Action (f_cmd, v_cmd) = ({f_cmd}, {v_cmd})")

        # Step env (example; adapt to your API)
        freq_index, speed, mec_power, com_power, thr, target_thr = env.get_sim_current_state(f_cmd, v_cmd)
        next_state = np.array([freq_index, speed, mec_power, com_power, thr, target_thr], dtype=np.float32)
        delta_soc, delta_soh = env.get_battery_delta_soc(mec_power, com_power)
        reward = env.get_reward(thr, delta_soc, delta_soh, target_thr, speed)
        done = 0  # or your terminal condition

        # Store transition with continuous action
        agent.store(state, a_cont, reward, next_state, done)

        # Update SAC
        log = agent.update()

        # Record reward in CSV
        writer.writerow([t, reward])

        state = next_state

        if t % 1000 == 0:
            agent.save("./save_models/")

print("[Training Complete]")
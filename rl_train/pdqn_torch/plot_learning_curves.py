import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def safe_convert(x):
    if isinstance(x, str):
        return float(x.strip('[]'))
    else:
        return float(x)

reward_runs = [pd.read_csv(f'./training_results/reward_{i}.csv', header=None).iloc[:, 1].apply(safe_convert) for i in range(2)]
loss_runs = [pd.read_csv(f'./training_results/loss_{i}.csv', header=None).iloc[:, 1].apply(safe_convert) for i in range(2)]
avgq_runs = [pd.read_csv(f'./training_results/maxq_{i}.csv', header=None).iloc[:, 1].apply(safe_convert) for i in range(2)]

# Stack into numpy arrays
reward_runs = np.stack(reward_runs)
loss_runs = np.stack(loss_runs)
avgq_runs = np.stack(avgq_runs)

# Plot reward
mean_rewards = np.mean(reward_runs, axis=0)
std_rewards = np.std(reward_runs, axis=0)

timesteps = np.arange(len(mean_rewards))

plt.figure()
plt.plot(timesteps, mean_rewards, label="Reward (mean)")
plt.fill_between(timesteps, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.3)
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.title('Reward Learning Curve')
plt.grid()
plt.legend()
plt.show()

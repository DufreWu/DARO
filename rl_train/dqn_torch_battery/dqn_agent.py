import os
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from network_torch import QNetwork
from robot_env import RobotEnv
import csv
import math

# Define CPU and GPU frequency lists
cpu_fre_list = [499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200]
speed_list = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]  # Speed

# Hyperparameters
experiment_time = 2000
target_thr = 25

# Define the DQNAgent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Action parameter list
        self.action_list = [(i, j) for i in range(len(cpu_fre_list)) for j in range(len(speed_list))]

        # Hyperparameters
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.1
        self.batch_size = 64
        self.train_start = 100
        self.q_max=0
        self.avg_q_max=0
        self.currentLoss=0
        self.learning_rate = 0.005
        self.memory = deque(maxlen=500)

        # Actor and Q-network models
        # self.actor_model = ParamActor(state_size, action_parameter_size).to(torch.device("cpu"))
        # self.target_actor_model = ParamActor(state_size, action_parameter_size).to(torch.device("cpu"))
        self.q_model = QNetwork(state_size, action_size).to(torch.device("cpu"))
        self.target_q_model = QNetwork(state_size, action_size).to(torch.device("cpu"))

        # Optimizers
        # self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=0.0005)
        self.q_optimizer = optim.Adam(self.q_model.parameters(), lr=0.0005)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Update target models
        self.update_target_model()

    def update_target_model(self):
        self.target_q_model.load_state_dict(self.q_model.state_dict())

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            print('get_action()-state = {}'.format(state))
            agent_input_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_model(agent_input_tensor).detach().numpy()[0]
            action = np.argmax(q_values)

        return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        agent_inputs = np.zeros((self.batch_size, self.state_size))
        agent_next_inputs = np.zeros((self.batch_size, self.state_size))

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

            # next_action_paras[i] = self.target_actor_model(torch.tensor([next_states[i]], dtype=torch.float32)).detach().numpy()
            
            for j in range(len(states[i])):
                agent_inputs[i][j]=states[i][j]
                agent_next_inputs[i][j]=next_states[i][j]

        with torch.no_grad():
            # Predict Q-values
            # pred_next_action_paras = self.target_actor_model(torch.tensor(next_states, dtype=torch.float32))
            pred_q_values = self.target_q_model(torch.tensor(agent_inputs, dtype=torch.float32))
            Qprime = torch.max(pred_q_values, 1, keepdim=True)[0].squeeze()

            # Compute the TD error
            rewards = torch.tensor(rewards, dtype=torch.float32) 
            target = rewards + self.discount_factor * Qprime

        # Compute current Q-values using policy network
        q_values = self.q_model(torch.tensor(agent_inputs, dtype=torch.float32))
        y_predicted = q_values.gather(1, torch.tensor(actions).view(-1, 1)).squeeze()
        y_expected = torch.tensor(target, dtype=torch.float32)
        loss_Q = self.loss_fn(y_predicted, y_expected)

        self.q_optimizer.zero_grad()
        loss_Q.backward()

        # Optimize actor
        # with torch.no_grad():
            # action_params = self.actor_model(torch.tensor(states, dtype=torch.float32))
        # action_params.require_grad = True
        Q = self.q_model(torch.tensor([states], dtype=torch.float32))
        Q_val = Q
        Q_loss = torch.mean(torch.sum(Q_val, dim=1))
        self.q_model.zero_grad()
        Q_loss.backward()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

state_size = 6
action_size = len(cpu_fre_list)*len(speed_list)

if __name__=="__main__":
    
    agent = DQNAgent(state_size, action_size = len(cpu_fre_list)*len(speed_list))
    env = RobotEnv()
    scores, episodes = [], []

    t=0
    learn=1
    ts=[]
    thr_data=[]
    power_data=[]
    speed_data=[]
    avg_q_max_data=[]
    loss_data = []
    reward_tmp=[]
    avg_reward=[]
    cnt=0
    freq_index=0
    speed_index=0
    state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    score=0
    action=0

    try:
        while t < experiment_time:       
            [freq_index, speed, mec_power, com_power, thr, thr_ref] = env.get_sim_current_state(freq_index, speed_list[speed_index])

            ts.append(t)
            thr_data.append(thr)
            power_data.append((com_power+mec_power)/speed_list[speed_index])
            speed_data.append(speed_list[speed_index])

            agent_input=[]
            next_state=(freq_index, speed, mec_power, com_power, thr, thr_ref)
            for i in range(len(next_state)):
                agent_input.append(next_state[i])
            agent_output = agent.q_model(torch.tensor([agent_input], dtype=torch.float32))

            agent.q_max += torch.max(agent_output).item()
            if t > 0:
                agent.avg_q_max = agent.q_max / t
            else:
                agent.avg_q_max = 0  # Handle the case where t is zero
            avg_q_max_data.append(agent.avg_q_max)
            loss_data.append(agent.currentLoss)

            # Pop dummy value at the first sensing
            if (com_power+mec_power<=0):
                com_power=20
                mec_power=13
            # e_d = (com_power+mec_power) / speed_list[speed_index]
            delta_soc, delta_soh = env.get_battery_delta_soc(mec_power, com_power)
            reward = env.get_reward(thr, delta_soc, delta_soh, target_thr, speed_list[speed_index])
            reward_tmp.append(reward)
            print('main()-reward = {}'.format(reward))
            if(len(reward_tmp)>=300) :
                reward_tmp.pop(0)

            done = 1
            # replay memory
            agent.append_sample(state, action, reward, next_state, done)

            print('[{}] state:{} action:{} speed:{} next_state:{} reward:{} thr:{}, avg_q={}'.format(
                t, state,action,speed_list[speed_index],next_state, reward, thr, agent.avg_q_max))
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward

            avg_reward.append(sum(reward_tmp) / len(reward_tmp))

            # print('learning_rate:{}'.format(agent.learning_rate))
            # get action
            state=next_state
            print('main()-state:{}'.format(state))

            action=agent.get_action(state)
            freq_index=agent.action_list[action][0]
            speed_index = agent.action_list[action][1]

            print('main()-after get_action()-speed:{}'.format(speed_list[speed_index]))            
            if done:
                agent.update_target_model()
            t = t+1
            print('experiment-time={}'.format(t))
            if t%500 == 0:
                scripted_q_model = torch.jit.script(agent.q_model)
                scripted_q_model.save("./save_model/dqn_model.pt")
                # scripted_actor_model = torch.jit.script(agent.actor_model)
                # scripted_actor_model.save("./save_model/model_actor_torch.pt")
                # torch.save(agent.q_model.state_dict(), "./save_model/model_Q_torch.pt")
                # torch.save(agent.actor_model.state_dict(), "./save_model/model_actor_torch.pt")
                print("[Save model]")
            if t==experiment_time:
                break
        # np.save(f'./training_results/reward_{index}.npy', np.array(avg_reward))
        # np.save(f'./training_results/qmax_{index}.npy', np.array(avg_q_max_data))
        # np.save(f'./training_results/loss_{index}.npy', np.array(loss_data))

    finally:
        print('[Save csv file]')
        ts = range(0, len(avg_q_max_data))

        f = open('./training_results/reward.csv', 'w', encoding='utf-8', newline='')
        f.write("ITERATION,REWARD\n") 
        wr = csv.writer(f)
        for row in zip(ts, avg_reward):  # Combine ts and avg_reward row by row
            wr.writerow(row)
        f.close()


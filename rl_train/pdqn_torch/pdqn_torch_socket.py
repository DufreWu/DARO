import os
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from network_torch import ParamActor, QNetwork
# from robot_env import RobotEnv
import csv
import math
import socket

PORT = 8702

# Define CPU and GPU frequency lists
cpu_fre_list = [499200, 652800, 806400, 960000, 1113600, 1267200, 1420800, 1574400, 1728000, 1881600, 2035200]
gpu_fre_list = [229500, 318750, 408000, 497250, 586500, 675750, 765000, 854250, 943500, 1032750, 1122000, 1211250, 1300500]

# Hyperparameters
speed_min = 20
speed_max = 50
experiment_time = 2000
target_fps = 25
beta = 8
loop_num = 3

# Define the PDQNAgent
class PDQNAgent:
    def __init__(self, state_size, action_size, action_parameter_size):
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.action_parameter_min = speed_min
        self.action_parameter_max = speed_max
        self.action_parameter_range = self.action_parameter_max - self.action_parameter_min

        # Action parameter list
        self.clk_action_list = [(i, j) for i in range(len(cpu_fre_list)) for j in range(len(gpu_fre_list))]

        # Hyperparameters
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.1
        self.batch_size = 64
        self.train_start = 200
        self.q_max=0
        self.avg_q_max=0
        self.currentLoss=0
        self.learning_rate = 1
        self.memory = deque(maxlen=500)

        # Actor and Q-network models
        self.actor_model = ParamActor(state_size, action_parameter_size).to(torch.device("cpu"))
        self.target_actor_model = ParamActor(state_size, action_parameter_size).to(torch.device("cpu"))
        self.q_model = QNetwork(state_size + action_parameter_size, action_size).to(torch.device("cpu"))
        self.target_q_model = QNetwork(state_size + action_parameter_size, action_size).to(torch.device("cpu"))

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=0.0005)
        self.q_optimizer = optim.Adam(self.q_model.parameters(), lr=0.0005)

        # Loss function
        self.loss_fn = nn.MSELoss()

        # Initialize target models
        self.update_target_model()

        # socket
        self.server_socket = 0

    def update_target_model(self):
        self.target_actor_model.load_state_dict(self.actor_model.state_dict())
        self.target_q_model.load_state_dict(self.q_model.state_dict())

    def get_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_parameter = self.actor_model(state_tensor).detach().numpy()[0]

        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            action_parameter = random.uniform(0, 1)
        else:
            print('get_action()-state = {}, action_parameter = {}'.format(state, action_parameter))
            agent_input = np.concatenate([state, np.array(action_parameter).flatten()]).astype(np.float32)
            agent_input_tensor = torch.FloatTensor(agent_input).unsqueeze(0)
            q_values = self.q_model(agent_input_tensor).detach().numpy()[0]
            action = np.argmax(q_values)

        return action, action_parameter

    def append_sample(self, state, action, action_para, reward, next_state, next_action_para, done):
        self.memory.append((state, action, action_para, reward, next_state, next_action_para, done))

    def train_model(self):
        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, action_paras, next_action_paras, rewards, dones = [], [], [], [], []
        agent_inputs = np.zeros((self.batch_size, self.state_size+self.action_parameter_size))
        agent_next_inputs = np.zeros((self.batch_size, self.state_size+self.action_parameter_size))

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            action_paras.append(mini_batch[i][2])
            rewards.append(mini_batch[i][3])
            next_states[i] = mini_batch[i][4]
            next_action_paras.append(mini_batch[i][5])
            dones.append(mini_batch[i][6])

            next_action_paras[i] = self.target_actor_model(torch.tensor([next_states[i]], dtype=torch.float32)).detach().numpy()
            
            for j in range(len(states[i])):
                agent_inputs[i][j]=states[i][j]
                agent_next_inputs[i][j]=next_states[i][j]
            agent_inputs[i][5]=action_paras[i]
            agent_next_inputs[i][5]=next_action_paras[i]

        # Predict Q-values
        q_values = self.q_model(torch.tensor(agent_inputs, dtype=torch.float32))
        next_q_values = self.q_model(torch.tensor(agent_next_inputs, dtype=torch.float32))

        # Compute targets
        targets = q_values.clone()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = float(rewards[i])
            else:
                targets[i, actions[i]] = rewards[i] + self.discount_factor * torch.max(next_q_values[i])

        # Compute loss and update Q-network
        self.q_optimizer.zero_grad()
        loss = self.loss_fn(q_values, targets)
        loss.backward()
        self.q_optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def create_server_socket(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("", 8702))
        self.server_socket.listen(5)
        print("Waiting connection")

state_size = 5
action_size = len(cpu_fre_list) * len(gpu_fre_list)
action_parameter_size = 1

def get_reward(fps, e_d, target_fps):
        print('get_reward()-e_d = {}'.format(e_d))
        v1 = 0.2
        u = max(1, fps / target_fps)

        if fps >= target_fps:
            u = 1
        else:
            u = fps / target_fps
        return u + v1 + beta / e_d

if __name__=="__main__":
    index = 0
    
    agent = PDQNAgent(5, len(cpu_fre_list)*len(gpu_fre_list), 1)
    scores, episodes = [], []

    t=0
    learn=1
    ts=[]
    fps_data=[]
    power_data=[]
    speed_data=[]
    avg_q_max_data=[]
    loss_data = []
    reward_tmp=[]
    avg_reward=[]
    cnt=0
    c_c=10
    g_c=12
    speed=5
    state=(11,12,20,27,30)
    action_para=0.0
    score=0
    action=0
    copy=1
    clk=11

    try:
        agent.create_server_socket()
        client_socket, address = agent.server_socket.accept()
        while t < experiment_time:
            msg = client_socket.recv(1024).decode('utf-8')
            state_tmp = msg.split(',')
            print('connection')
            if not msg:
                print('No receiveddata')
                continue
            
            c_c=int(state_tmp[0])
            g_c=int(state_tmp[1])
            c_p=float(state_tmp[2])
            m_p=float(state_tmp[3])
            fps=float(state_tmp[4])

            ts.append(t)
            fps_data.append(fps)
            power_data.append((c_p+m_p))
            speed_data.append(speed)

            agent_input=[]
            next_state=(c_c, g_c, c_p, m_p, fps)
            print('main()-next_state = {}'.format(next_state))
            next_action_para=speed
            print('main()-next_action_para = {}'.format(next_action_para))
            for i in range(len(next_state)):
                agent_input.append(next_state[i])
            agent_input.append(next_action_para)
            print('main()-agent_input = {}'.format(agent_input))
            agent_output = agent.q_model(torch.tensor([agent_input], dtype=torch.float32))

            agent.q_max += torch.max(agent_output).item()
            if t > 0:
                agent.avg_q_max = agent.q_max / t
            else:
                agent.avg_q_max = 0  # Handle the case where t is zero
            avg_q_max_data.append(agent.avg_q_max)
            loss_data.append(agent.currentLoss)

            # Pop dummy value at the first sensing
            if (c_p+m_p<=0):
                c_p=20
                m_p=13
            e_d = (c_p/1000+m_p/1000) / (speed/10)
            reward = get_reward(fps, e_d, target_fps)
            reward_tmp.append(reward)
            if(len(reward_tmp)>=300) :
                reward_tmp.pop(0)

            done = 1
            # replay memory
            agent.append_sample(state, action, speed, reward, next_state, next_action_para, done)
            # double copy for highlighted sample
            for i in range(copy):
                if reward<0:
                    agent.learning_rate = 1
                    agent.append_sample(state, action, speed, reward, next_state, next_action_para, done)
                    agent.append_sample(state, action, speed, reward, next_state, next_action_para, done)
                if reward>1:
                    agent.learning_rate = 1
                    agent.append_sample(state, action, speed, reward, next_state, next_action_para, done)
                    agent.append_sample(state, action, speed, reward, next_state, next_action_para, done)

            print('[{}] state:{} action:{} speed:{} next_state:{} next_action_para:{} reward:{} fps:{}, avg_q={}'.format(
                t, state,action,speed,next_state,next_action_para,reward,fps,agent.avg_q_max))
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            score += reward

            avg_reward.append(sum(reward_tmp) / len(reward_tmp))

            print('learning_rate:{}'.format(agent.learning_rate))
            # get action
            state=next_state
            speed=next_action_para
            print('main()-state:{}'.format(state))

            if fps<target_fps:
                if np.random.rand() <= 0.3:
                    print('previous clock : {} {}'.format(c_c,g_c))
                    c_c=random.randint(0, len(cpu_fre_list)-1)
                    g_c=random.randint(0, len(gpu_fre_list)-1)
                    print('explore higher clock@@@@@  {} {}'.format(c_c, g_c))
                    action=c_c*len(gpu_fre_list)+g_c
                    speed=random.randint(speed_min, speed_max)
                else:
                    action, action_para=agent.get_action(state)
                    c_c=agent.clk_action_list[action][0]
                    g_c=agent.clk_action_list[action][1]
                    speed = (action_para + 1) * (speed_max - speed_min) / 2 + speed_min
            else:
                action, action_para=agent.get_action(state)
                c_c=agent.clk_action_list[action][0]
                g_c=agent.clk_action_list[action][1]
                speed = (action_para + 1) * (speed_max - speed_min) / 2 + speed_min

            print('main()-speed:{}-cpu_freq:{}-gpu_freq:{}'.format(speed, cpu_fre_list[c_c], gpu_fre_list[g_c]))
            send_msg=str(int(speed))+','+str(c_c)+','+str(g_c)
            client_socket.send(send_msg.encode())
            print('send msg:{}'.format(send_msg))
            if done:
                agent.update_target_model()
            t = t+1
            print('experiment-time={}'.format(t))
            if t%60 == 0:
                agent.learning_rate=0.2
                print('[Reset learning_rate]')
            if t%100 == 0:
                scripted_q_model = torch.jit.script(agent.q_model)
                scripted_q_model.save("./save_model/model_Q_torch.pt")
                scripted_actor_model = torch.jit.script(agent.actor_model)
                scripted_actor_model.save("./save_model/model_actor_torch.pt")
                # torch.save(agent.q_model.state_dict(), "./save_model/model_Q_torch.pt")
                # torch.save(agent.actor_model.state_dict(), "./save_model/model_actor_torch.pt")
                print("[Save model]")
            if t==experiment_time:
                index = index + 1
                break

    finally:
        print('[Save csv file]:{}'.format(index))
        ts = range(0, len(avg_q_max_data))

        f = open(f'./training_results/maxq_{index}.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        for row in zip(ts, avg_q_max_data):  # Combine ts and avg_q_max_data row by row
            wr.writerow(row)
        f.close()

        f = open(f'./training_results/reward_{index}.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        for row in zip(ts, avg_reward):  # Combine ts and avg_reward row by row
            wr.writerow(row)
        f.close()

        f = open(f'./training_results/loss_{index}.csv', 'w', encoding='utf-8', newline='')
        wr = csv.writer(f)
        for row in zip(ts, loss_data):  # Combine ts and loss_data row by row
            wr.writerow(row)
        f.close()
        agent.server_socket.close()
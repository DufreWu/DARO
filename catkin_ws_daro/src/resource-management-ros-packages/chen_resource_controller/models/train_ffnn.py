#!/usr/bin/env python

from cmath import nan
import os
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import sys
from datetime import datetime

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        # self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_feature, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Linear(32, n_output)
        )

    def forward(self, x):
        return self.layers(x)


class EnergyThrouputDataset(Dataset):    
    def __init__(self, exp_dir):
        sys_data = pd.read_csv(os.path.join(exp_dir, "sys.csv"), delimiter=";")
        app_data = pd.read_csv(os.path.join(exp_dir, "appls.csv"), delimiter=";", )
        fa_harris_data = pd.read_csv(os.path.join(exp_dir, "fa_harris_stat.csv"), delimiter=";")

        print("Data read from {}: sys [{}], app [{}], fa_harris [{}]".format(exp_dir, len(sys_data), len(app_data), len(fa_harris_data)))

        input_data = []
        output_data = []
        for i in range(len(sys_data) - 1):
            sys_row = sys_data.iloc[i]
            app_row = app_data.iloc[i]
            try:
                next_sys_row = sys_data.iloc[i+1]
                next_app_row = app_data.iloc[i+1]
            except:
                print("faild to read line {} from data in {}".format(i, exp_dir))
                continue
            if pd.isnull(app_row["PID"]) or \
                    pd.isnull(next_app_row["PID"]) or \
                        app_row["TH_CUR"] == 0 or next_app_row["TH_CUR"] == 0:
                continue

            input_data.append([
                sys_row["FREQ_L"], 
                sys_row["UTIL0"], 
                sys_row["UTIL1"], 
                sys_row["UTIL2"], 
                sys_row["UTIL3"], 
                sys_row["UTIL4"], 
                sys_row["UTIL5"], 
                sys_row["SPEED"], 
                app_row["TH_CUR"], 
                (sys_row["POWER"]+sys_row["MECHW"])/sys_row["SPEED"],
                next_sys_row["FREQ_L"], next_sys_row["SPEED"]
            ])
            output_data.append([
                next_app_row["TH_CUR"], 
                (next_sys_row["POWER"]+next_sys_row["MECHW"])/next_sys_row["SPEED"],       
            ])

        print(len(input_data), len(output_data))
        self.x = torch.from_numpy(np.array(input_data, dtype=np.float32))
        self.y = torch.from_numpy(np.array(output_data, dtype=np.float32))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        dt = {'x': self.x[idx], 'y':self.y[idx]}
        # print(dt)
        return dt

class ControllerModel:
    def __init__(self, exp_dir, out_dir, n_hidden):
        self.input_names = ["freq_cur", "util0","util0","util0","util0","util0","util0",
                     "speed_cur", "thr_cur", "energy_curr", "freq_next", "speed_next"]
        self.output_names = ["thr_next", "energy_next"]
        
        self.last_train_time = 0
        
        self.net = Net(len(self.input_names), n_hidden, len(self.output_names))
        

        self.exp_dir = exp_dir
        self.out_dir = out_dir
        
        
    def train(self, test_set_dir, lr=3e-2, epochs=1000, batch_size=32):
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(self.net)  # net architecture
        self.log_file_train = os.path.join(self.out_dir, "model_log_train.csv")
        self.log_file_test = os.path.join(self.out_dir, "model_log_test.csv")
        self.log_file_test_predictions = os.path.join(self.out_dir, "model_log_test_predictions.csv")
        
        if not os.path.exists(self.log_file_train):
            with open(self.log_file_train, "w") as f:
                print("epoch,i_batch,loss_batch", file=f)
        
        if not os.path.exists(self.log_file_test):
            with open(self.log_file_test, "w") as f:
                print("epoch,loss_test,mae_test,r2_test", file=f)
        
        with open(self.log_file_test_predictions, "w") as f:
            print(','.join(self.input_names+self.output_names+["thr_next_pred", "energy_next_pred"]), file=f)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
        energy_dataset = EnergyThrouputDataset(self.exp_dir)

        dataloader = DataLoader(energy_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)
        print("Time: {}. Training on {} data points".format(now, len(energy_dataset)))
        if test_set_dir is not None:
            testset = EnergyThrouputDataset(test_set_dir)
            # print(testset[:][x])
            x = testset[:]["x"]
            y = testset[:]["y"]
        
        lftr_out = open(self.log_file_train, mode="a")
        lfts_out = open(self.log_file_test, mode="a")
        lftsp_out = open(self.log_file_test_predictions, mode="a")

        with tqdm(range(epochs)) as pbar:
            for t in pbar:
                for i_batch, sample in enumerate(dataloader):
                    x_batch, y_batch = sample['x'], sample['y']
                    # print(x_batch.shape)
                    prediction = self.net(x_batch)     # input x and predict based on x
                    loss = loss_func(prediction, y_batch)     # must be (1. nn output, 2. target)

                    optimizer.zero_grad()   # clear gradients for next train
                    loss.backward()         # backpropagation, compute gradients
                    optimizer.step()        # apply gradients
                    # pbar.set_description("iter: \t {} \t batch loss: \t {} \t".format(t, loss.item()))
                    print(','.join([str(x) for x in [t, i_batch, loss.item()]]), file=lftr_out)
                
                if test_set_dir is not None:
                    y_pred = self.net(x)
                    loss = loss_func(y, y_pred)
                    loss = loss.item()
                    y_pred = y_pred.detach().numpy()
                    mae = mean_absolute_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    print(','.join([str(x) for x in [t, loss, mae, r2]]), file=lfts_out)
            
        for i in range(len(x)):
            inp = x[i].detach().numpy()
            outp = y[i].detach().numpy()
            pred = y_pred[i]
            dt = np.concatenate((inp, outp, pred))
            print(",".join([str(d) for d in dt]), file=lftsp_out)

        # for i in range(len(y_pred)):
        #     print("input: ")
        #     print(sample["x"][i].detach().numpy())
        #     print("true output:")
        #     print(sample["y"][i].detach().numpy())
        #     print("perdiction:")
        #     print(y_pred[i])
        #     print("----------------")
        # print("NN test error: \t\t: {}".format(mean_absolute_error(sample["y"].detach().numpy(), y_pred)))
        # print("NN test r2 score \t: {}".format(r2_score(sample["y"].detach().numpy(), y_pred)))
        # print()

        lftr_out.close()
        lfts_out.close()
        lftsp_out.close()

def save_controller(mdl, fp):
    with open(fp, 'wb') as outp:
        pickle.dump(mdl, outp, pickle.HIGHEST_PROTOCOL)


def load_controller(fp):
    with open(fp, 'rb') as inp:
        controller_model = pickle.load(inp)
        return controller_model


def train_model(args):
    os.makedirs(args.out_dir, exist_ok=True)
    
    dump_file = os.path.join(args.out_dir, "train.dump")
    if os.path.exists(dump_file):
        controller_model = load_controller(dump_file)
    else:
        controller_model = ControllerModel(args.exp_dir, args.out_dir, n_hidden=32)
    
    controller_model.train(args.test_set_dir, epochs=args.epochs, lr=args.lr)
    
    save_controller(controller_model, dump_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot results exp directory')
    parser.add_argument('exp_dir', type=str, help='path to the exp dir')
    parser.add_argument('out_dir', type=str, help='path to the output (model) dir')
    parser.add_argument('--test_set_dir', type=str, default=None, help='path to the test set dir')
    parser.add_argument('--epochs', type=int, 
                    default=100, help='epochs to train')
    parser.add_argument('--lr', type=float, 
                    default=1e-3, help='learning rate')
    args = parser.parse_args()
    
    print(' '.join(sys.argv))

    train_model(args)

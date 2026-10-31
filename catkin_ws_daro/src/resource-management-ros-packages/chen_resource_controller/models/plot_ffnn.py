#!/usr/bin/env python

import argparse
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
import sys

font = {'size'   : 14}
matplotlib.rc('font', **font)

def plot_multi(data, keys_dict, outpath):
    print(keys_dict)
    fig, axs = plt.subplots(len(keys_dict), figsize=(10,len(keys_dict)*3))
    if len(keys_dict) == 1:
        axs = [axs]
    
    for i,name in enumerate(keys_dict):
        for k in keys_dict[name]:
            axs[i].plot(data["x_ax"], data[k], label=k)
        axs[i].set_ylabel(name)
        lgd = axs[i].legend(ncol=len(keys_dict[name]), loc='lower center', bbox_to_anchor=(0.5, 1.02)) #bbox_to_anchor=(1.05, 1.0), loc='upper left'
    plt.subplots_adjust(hspace=1)
    plt.savefig(outpath, bbox_extra_artists=(lgd,)) #, , bbox_inches='tight'  


def plot_results(model_dir, out_dir):
    train_data = pd.read_csv(os.path.join(model_dir, "model_log_train.csv"), delimiter=",")
    test_data = pd.read_csv(os.path.join(model_dir, "model_log_test.csv"), delimiter=",")
    test_data_predictions = pd.read_csv(os.path.join(model_dir, "model_log_test_predictions.csv"), delimiter=",")
    
    print(train_data.head())
    print(max(train_data["i_batch"]))
    train_data["x_ax"] = (train_data["i_batch"]+1).cumsum()
    test_data["x_ax"] = (test_data["epoch"]+1).cumsum()
    
    outpath = os.path.join(out_dir, "model_train_plot.pdf")
    
    fig, axs = plt.subplots(4, figsize=(10,12))
    axs[0].plot(train_data["x_ax"], train_data["loss_batch"], label="train_loss_batch")
    axs[0].set_ylabel("train_loss_batch")
    axs[0].plot(train_data["x_ax"].iloc[-1], train_data["loss_batch"].iloc[-1], 
        marker="o", markersize=5, markeredgecolor="red",
        markerfacecolor="green")
    axs[0].annotate("{:.2f}".format(train_data["loss_batch"].iloc[-1]), 
        (train_data["x_ax"].iloc[-1], train_data["loss_batch"].iloc[-1]))
        

    axs[1].plot(test_data["x_ax"], test_data["loss_test"], label="test_loss")
    axs[1].set_ylabel("test_loss")
    axs[1].plot(test_data["x_ax"].iloc[-1], test_data["loss_test"].iloc[-1], 
        marker="o", markersize=5, markeredgecolor="red",
        markerfacecolor="green")
    axs[1].annotate("{:.2f}".format(test_data["loss_test"].iloc[-1]), 
        (test_data["x_ax"].iloc[-1], test_data["loss_test"].iloc[-1]))
    

    axs[2].plot(test_data["x_ax"], test_data["mae_test"], label="mae_test")
    axs[2].set_ylabel("mae_test")
    axs[2].plot(test_data["x_ax"].iloc[-1], test_data["mae_test"].iloc[-1], 
        marker="o", markersize=5, markeredgecolor="red",
        markerfacecolor="green")
    axs[2].annotate("{:.2f}".format(test_data["mae_test"].iloc[-1]), 
        (test_data["x_ax"].iloc[-1], test_data["mae_test"].iloc[-1]))
    

    axs[3].plot(test_data["x_ax"], test_data["r2_test"], label="r2_test")
    axs[3].set_ylabel("r2_test")
    axs[3].plot(test_data["x_ax"].iloc[-1], test_data["r2_test"].iloc[-1], 
        marker="o", markersize=5, markeredgecolor="red",
        markerfacecolor="green")
    axs[3].annotate("{:.2f}".format(test_data["r2_test"].iloc[-1]), (test_data["x_ax"].iloc[-1], test_data["r2_test"].iloc[-1]))
    
    plt.subplots_adjust(hspace=1)
    plt.savefig(outpath) #, , bbox_inches='tight'  

    
    outpath = os.path.join(out_dir, "model_predictions_plot.pdf")
    
    fig, axs = plt.subplots(2, figsize=(10,12))
    print(len(test_data_predictions))
    print(len(test_data_predictions["thr_next"]))
    axs[0].plot(range(len(test_data_predictions)), test_data_predictions["thr_next"], label="actual_trh")
    axs[0].plot(range(len(test_data_predictions)), test_data_predictions["thr_next_pred"], label="pred_trh")
    axs[0].set_ylabel("thr")
    axs[0].legend()

    axs[1].plot(range(len(test_data_predictions)), test_data_predictions["energy_next"], label="actual_energy")
    axs[1].plot(range(len(test_data_predictions)), test_data_predictions["energy_next_pred"], label="pred_energy")
    axs[1].set_ylabel("energy")
    axs[1].legend()

    plt.subplots_adjust(hspace=1)
    plt.savefig(outpath)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot progress of a model trainig directory')
    parser.add_argument('model_dir', type=str, help='path to the model dir')
    
    args = parser.parse_args()
    print(' '.join(sys.argv))

    out_dir = os.path.join(args.model_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    plot_results(args.model_dir, out_dir)
import os
import random
import numpy as np
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from LoadSyntheticdata import Synthetic_Dataset
from torch.utils import data
from FID import compute_fid


def load_synthetic_data():
    syn_data = Synthetic_Dataset()
    list_syn_data = [list(item) for item in syn_data]
    return list_syn_data

def categorize_data(list_syn_data):
    l_syn = []
    l_real = []

    l_running_real = []
    l_running_syn = []

    l_walking_real = []
    l_walking_syn = []

    l_GoingDownS_real = []
    l_GoingDownS_syn = []

    colors = []

    for item in list_syn_data:
        l_syn.append(item[0])
        l_real.append(item[2])

        if torch.all(torch.eq(item[1], torch.tensor([[0., 1., 0.]]))):
            l_running_real.append(item[2])
            l_running_syn.append(item[0])
            colors.append("running")

        if torch.all(torch.eq(item[1], torch.tensor([[1., 0., 0.]]))):
            l_walking_real.append(item[2])
            l_walking_syn.append(item[0])
            colors.append("walking")

        if torch.all(torch.eq(item[1], torch.tensor([[0., 0., 1.]]))):
            l_GoingDownS_real.append(item[2])
            l_GoingDownS_syn.append(item[0])
            colors.append("Going_Down_Stairs")

    return l_syn, l_real, l_running_real, l_running_syn, l_walking_real, l_walking_syn, l_GoingDownS_real, l_GoingDownS_syn, colors

def reshape_and_normalize(l_syn):
    l_syn = np.array(l_syn).reshape(-1, 3, 1, 150)
    l_syn = normalization(l_syn)
    l_syn = l_syn.reshape(-1, 3, 150)
    return l_syn

def _normalize(epoch):
    e = 1e-10
    result = (epoch - epoch.mean(axis=0)) / (np.sqrt(epoch.var(axis=0)) + e)
    return result

def normalization(epochs):
    for i in range(epochs.shape[0]):
        for j in range(epochs.shape[1]):
            epochs[i, j, 0, :] = _normalize(epochs[i, j, 0, :])
    return epochs

def preprocess_data(l_real, l_syn):
    l_real_p = [np.array([*list(sig[channel]), 0]) for sig in l_real for channel in range(3)]
    l_real_p = np.array(l_real_p).reshape(-1, 3, 151)
    l_real_p = np.transpose(l_real_p, (0, 2, 1))
    l_syn_p = [np.array([*list(sig[channel]), 0]) for sig in l_syn for channel in range(3)]
    l_syn_p = np.array(l_syn_p).reshape(-1, 3, 151)
    l_syn_p = np.transpose(l_syn_p, (0, 2, 1))

    return l_real_p, l_syn_p

def add_noise_to_syn_data(l_syn, target_noise_db):
    target_noise_watts = 10 ** (target_noise_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(target_noise_watts), size=l_syn.shape)
    noise_syn = l_syn + noise_volts
    return noise_syn

def preprocess_noise_data(noise_syn):
    noise_syn_p = [np.array([*list(sig[channel]), 0]) for sig in noise_syn for channel in range(3)]
    noise_syn_p = np.array(noise_syn_p).reshape(-1, 3, 151)
    noise_syn_p = np.transpose(noise_syn_p, (0, 2, 1))
    return noise_syn_p

def plot_real_data_samples(syn_data, save_path):
    fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    fig.suptitle('Real data samples of Walking activity without adding noise', fontsize=14)
    t = 0
    for i in range(0, 100):
        if torch.all(torch.eq(syn_data[i][1], torch.tensor([[1., 0., 0.]]))):
            axs[t].plot(syn_data[i][2][0][:], c="brown")
            axs[t].plot(syn_data[i][2][1][:], c="royalblue")
            axs[t].plot(syn_data[i][2][2][:], c="mediumseagreen")
            t += 1
            if t == 3:
                break
    plt.xlabel('Timesteps')
    plt.ylabel('Acceleration', horizontalalignment='right', y=3.5)
    plt.legend(['x', 'y', 'z'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(save_path, bbox_inches="tight")

def plot_noise_data_samples(syn_data, target_noise_db, save_path):
    target_noise_watts = 10 ** (target_noise_db / 10)
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(target_noise_watts), size=syn_data[0][2][0][:].shape)

    fig, axs = plt.subplots(3, 1, figsize=(5, 5))
    fig.suptitle(f'Real data samples of Walking activity with added noise (power = {target_noise_db})', fontsize=14)
    t = 0
    for i in range(0, 100):
        if torch.all(torch.eq(syn_data[i][1], torch.tensor([[1., 0., 0.]]))):
            x_volts = syn_data[i][2][0][:] + noise_volts
            y_volts = syn_data[i][2][1][:] + noise_volts
            z_volts = syn_data[i][2][2][:] + noise_volts
            axs[t].plot(x_volts, c="brown")
            axs[t].plot(y_volts, c="royalblue")
            axs[t].plot(z_volts, c="mediumseagreen")
            t += 1
            if t == 3:
                break
    plt.xlabel('Timesteps')
    plt.ylabel('Acceleration', horizontalalignment='right', y=3.5)
    plt.legend(['x', 'y', 'z'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(save_path, bbox_inches="tight")
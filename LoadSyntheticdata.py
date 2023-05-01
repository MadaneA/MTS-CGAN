# Made them to a Pytorch Dataset
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from dataLoader import *
from tqdm import tqdm
import torch
from MTSCGAN import *
import numpy as np
import random
import os

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

#change model path depending on the pre-trained-model : ALPHA09_Training/checkpoint_epochs_2600 for alpha 0.9 checkpoint_epochs_2600 - don't forget to change alpha_conv value in MTSCGAN.py
class Synthetic_Dataset(Dataset):
    def __init__(self,
                 signals_model_path = './pre-trained-models/ALPHA09_Training/checkpoint_epochs_2600',
                 sample_size = 300
                 ):

        self.sample_size = sample_size

        #Generate signals Data
        signals_gen_net = Generator() #seq_len=150, channels=1, latent_dim=100
        signals_ckp = torch.load(signals_model_path, map_location=torch.device('cpu'))
        signals_gen_net.load_state_dict(signals_ckp['gen_state_dict'])

        test_set = unimib_load_dataset(incl_xyz_accel = True, is_normalize = True, data_mode = 'Train')
        test_loader = data.DataLoader(test_set, batch_size=1, num_workers=1, shuffle = True, worker_init_fn=seed_worker, generator=g)
        
        real_signals = []
        syn_signals = []
        signals_label = []
        
        for iter_idx, (sig, context) in enumerate(tqdm(test_loader)):

            context_mat = context.type(torch.FloatTensor)
            
            real_signal = sig.type(torch.FloatTensor).detach().numpy()
            real_signal = real_signal.reshape(real_signal.shape[1], real_signal.shape[3])
            real_signals.append(real_signal)
            # Sample noise as generator input
            z = torch.FloatTensor(np.random.normal(0, 1, (context.shape[0], 100)))

            fake_sig = signals_gen_net(z, context_mat).detach().numpy()
            fake_sig = fake_sig.reshape(fake_sig.shape[1], fake_sig.shape[3])

            sig_label = context.type(torch.FloatTensor)
            signals_label.append(sig_label)
            syn_signals.append(fake_sig)

        real_signals = np.array(real_signals)
        syn_signals = np.array(syn_signals)
        signals_label = np.array(signals_label)

        self.syn_signals = syn_signals
        self.signals_label = signals_label
        self.real_signals = real_signals

        print("syn_signals", self.syn_signals.shape)
        print("signals_label", self.signals_label.shape)
        print("real_signals", self.real_signals.shape)


    def __len__(self):
        return self.sample_size #* 2

    def __getitem__(self, idx):
        return self.syn_signals[idx], self.signals_label[idx], self.real_signals[idx]

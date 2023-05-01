import os
import shutil 
from shutil import unpack_archive
import requests 
from scipy import io 
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.preprocessing import  OneHotEncoder 
from torch.utils.data import Dataset, DataLoader

class_dict = {'StandingUpFS':0,'StandingUpFL':1,'Walking':2,'Running':3,'GoingUpS':4,'Jumping':5,'GoingDownS':6,'LyingDownFS':7,'SittingDown':8}

class unimib_load_dataset(Dataset):
    def __init__(self,
        verbose = False,
        incl_xyz_accel = False, #include component accel_x/y/z in ____X data
        is_normalize = False,
        split_subj = dict
                    (train_subj = [4,5,6,7,8,10,11,12,14,15,19,20,21,22,24,26,27,29],
                    validation_subj = [1,9,16,23,25,28],
                    test_subj = [2,3,13,17,18,30]),
        data_mode = 'Train'): 

        self.verbose = verbose
        self.incl_xyz_accel = incl_xyz_accel
        self.split_subj = split_subj
        self.data_mode = data_mode       
        self.is_normalize = is_normalize      
        self.class_name = ['Walking','Running','GoingDownS'] 
        
        L = [class_dict[x] for x in self.class_name]
        L_resh = np.array(L).reshape(-1,1)
        # encode classes
        def encode_classes(data):
            ohe = OneHotEncoder(sparse = False)
            ohe.fit(data)
            data_enc = ohe.transform(data)
            return data_enc

        data_enc = encode_classes(L_resh)
        dict_enc = {'2':data_enc[0],'3':data_enc[1],'6':data_enc[2]}
        
        #Download and unzip original dataset
        if (not os.path.isfile('./UniMiB-SHAR.zip')):
            print("Downloading UniMiB-SHAR.zip file")
            #invoking the shell command fails when exported to .py file
            #redirect link https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            #!wget https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
            self.download_url('https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip','./UniMiB-SHAR.zip')
        if (not os.path.isdir('./UniMiB-SHAR')):
            shutil.unpack_archive('./UniMiB-SHAR.zip','.','zip')
        #Convert .mat files to numpy ndarrays
        path_in = './UniMiB-SHAR/data'
        #loadmat loads matlab files as dictionary, keys: header, version, globals, data
        adl_data = io.loadmat(path_in + '/adl_data.mat')['adl_data']
        adl_names = io.loadmat(path_in + '/adl_names.mat', chars_as_strings=True)['adl_names']
        adl_labels = io.loadmat(path_in + '/adl_labels.mat')['adl_labels']

        if(self.verbose):
            headers = ("Raw data","shape", "object type", "data type")
            mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                    ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                    ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
            print(tabulate(mydata, headers=headers))
        #Reshape data and compute total (rms) acceleration
        num_samples = 151
        #UniMiB SHAR has fixed size of 453 which is 151 accelX, 151 accely, 151 accelz
        adl_data = np.reshape(adl_data,(-1,num_samples,3), order='F') #uses Fortran order
        #remove component accel if needed
        if (not self.incl_xyz_accel):
            adl_data = np.delete(adl_data, [0,1,2], 2)
        if(verbose):
            headers = ("Reshaped data","shape", "object type", "data type")
            mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                    ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                    ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
            print(tabulate(mydata, headers=headers))
        #Split train/test sets, combine or make separate validation set
        #ref for this numpy gymnastics - find index of matching subject to sub_train/sub_test/sub_validate
        #https://numpy.org/doc/stable/reference/generated/numpy.isin.html

        act_num = (adl_labels[:,0])-1 #matlab source was 1 indexed, change to 0 indexed
        sub_num = (adl_labels[:,1]) #subject numbers are in column 1 of labels

        train_index = np.nonzero(np.isin(sub_num, self.split_subj['train_subj'] +
                                        self.split_subj['validation_subj']))
        x_train = adl_data[train_index]
        y_train = act_num[train_index]

        test_index = np.nonzero(np.isin(sub_num, self.split_subj['test_subj']))
        x_test = adl_data[test_index]
        y_test = act_num[test_index]

        if (verbose):
            print("x/y_train shape ",x_train.shape,y_train.shape)
            print("x/y_test shape  ",x_test.shape,y_test.shape)

        # reshape x_train, x_test data shape from (BH, length, channel) to (BH, channel, 1, length)
        self.x_train = np.transpose(x_train, (0, 2, 1))
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1, self.x_train.shape[2])
        self.x_train = self.x_train[:,:,:,:-1]
        self.y_train = y_train

        self.x_test = np.transpose(x_test, (0, 2, 1))
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1, self.x_test.shape[2])
        self.x_test = self.x_test[:,:,:,:-1]
        self.y_test = y_test
        
        print(f'x_train shape is {self.x_train.shape}, x_test shape is {self.x_test.shape}')
        print(f'y_train shape is {self.y_train.shape}, y_test shape is {self.y_test.shape}')

        if self.is_normalize:
            self.x_train = self.normalization(self.x_train)
            self.x_test = self.normalization(self.x_test)

        #Return the give class train/test data & labels
        one_class_train_data = []
        one_class_train_labels = []
        one_class_test_data = []
        one_class_test_labels = []

        for i, label in enumerate(y_train):
            if label in L:
                one_class_train_data.append(self.x_train[i])
                one_class_train_labels.append(dict_enc[str(label)])

        for i, label in enumerate(y_test):
            if label in L:
                one_class_test_data.append(self.x_test[i])
                one_class_test_labels.append(dict_enc[str(label)])
                
        self.one_class_train_data = np.array(one_class_train_data)
        self.one_class_train_labels = np.array(one_class_train_labels)
        self.one_class_test_data = np.array(one_class_test_data)
        self.one_class_test_labels = np.array(one_class_test_labels)

        print(f'return three classes data and labels {self.class_name}')
        print(f'train_data shape is {self.one_class_train_data.shape}')
        print(f'train label shape is {self.one_class_train_labels.shape}')

    def download_url(self, url, save_path, chunk_size=128):
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

    def _normalize(self, epoch):
        """ A helper method for the normalization method.
            Returns
                result: a normalized epoch
        """
        e = 1e-10
        result = (epoch - epoch.mean(axis=0)) / ((np.sqrt(epoch.var(axis=0)))+e)
        return result

    def normalization(self, epochs):
        """ Normalizes each epoch e s.t mean(e) = 0 and var(e) = 1
            Args:
                epochs - Numpy structure of epochs
            Returns:
                epochs_n - mne data structure of normalized epochs (mean=0, var=1)
        """
        for i in range(epochs.shape[0]):
            for j in range(epochs.shape[1]):
                epochs[i,j,0,:] = self._normalize(epochs[i,j,0,:])

        return epochs

    def __len__(self):
        if self.data_mode == 'Train':
            return len(self.one_class_train_labels)

        else:
            return len(self.one_class_test_labels)


    def __getitem__(self, idx):
        if self.data_mode == 'Train':
            return self.one_class_train_data[idx], self.one_class_train_labels[idx]

        else:
            return self.one_class_test_data[idx], self.one_class_test_labels[idx]


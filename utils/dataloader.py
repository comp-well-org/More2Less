from configs.opts import DATA_PATH
import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import normalize_data, DA_Jitter


class MyDataset(Dataset):
    def __init__(self, filename, is_training=True):
        super(MyDataset).__init__()
        self.filename = filename;
        self.is_training = is_training;


        # first, check whether or not the file exist
        # filename_data must not be none.
        if not os.path.isfile(filename):
            print(filename + "doesn't exist!\n")
            exit(0)
        # then load the data.
        data_dict = np.load(filename, allow_pickle=True).item();

        self.ecg = normalize_data(data_dict['seq_ecg'])
        self.fitbit = normalize_data(data_dict['seq_fitbit'])
        self.stress = data_dict['stress']
        self.anxiety = data_dict['anxiety']


        #  use stress or anxiety as target for detection.
        self.labels = np.expand_dims(self.stress, axis=1).astype(np.float)
        print(np.sum(self.labels))

    def __len__(self):
        return self.ecg.shape[0]


    def __getitem__(self, index):
        return self.ecg[index],  self.fitbit[index],  self.labels[index]




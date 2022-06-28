import numpy as np
import os
import random
import torch
from torch import optim, threshold
from torch.functional import norm
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn  as nn
from configs.opts import *
from utils.utils import normalize_data

from models.model import MyModel
from models.training import train_model
from utils.dataloader import MyDataset

# write hyper-parameters into files, so we can track the training 
# procedure later.
with open(MODEL_PATH + 'params.txt', 'w') as fd:
    fd.write("DATA_PATH = "+ DATA_PATH + '\n')
    fd.write("MODEL_PATH = "+ MODEL_PATH + '\n')
    fd.write("CUDA_DEVICE = "+ CUDA_DEVICE + '\n')
    fd.write("NUM_CLASS = " + str(NUM_CLASS) + '\n')
    fd.write("HIDDEN_DIM = " + str(HIDDEN_DIM) + '\n')
    fd.write("INPUT_LEN = " + str(INPUT_LEN) + '\n')
    fd.write("BATCH_SIZE = " + str(BATCH_SIZE) + '\n')
    fd.write("TRAINING_EPOCHES = " + str(TRAINING_EPOCHES) + '\n')
    fd.write("LEARNING_RATE = " + str(LEARNING_RATE) + '\n')
    fd.write("DECAY_RATE = " + str(DECAY_RATE) + '\n')

dataloaders={
    'train': DataLoader(MyDataset(DATA_PATH+ "data_train.npy", is_training=True), shuffle=True, batch_size=BATCH_SIZE), # [Batch_size, 7, 138], [Batch_size]
    #'val': DataLoader(MyDataset(DATA_PATH+ "data_val.npy", is_training=True), shuffle=True, batch_size=BATCH_SIZE), # [Batch_size, 7, 138], [Batch_size]
    'test':  DataLoader(MyDataset(DATA_PATH+ "data_test.npy", is_training=False),  shuffle=False, batch_size=BATCH_SIZE)
}

# define a random seed to keep the training-testing stable.
torch.manual_seed(10000)
# define the model.
model = MyModel(INPUT_DIM_M1, INPUT_DIM_M2, INPUT_LEN, hidden_dim=64, num_classes=NUM_CLASS)
# detect if we have any GPU support.
# or a specifc GPU is available.
device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else "cpu")
# send model to GPU
model = model.to(device)

params_to_update = model.parameters()
print("Params to learn: ")
for name, param in model.named_parameters():
    if param.requires_grad == True:
        print("\t", name)

# define an optimizer.
#optimizer_ft_m1 = optim.Adam([param for name, param in model.named_parameters() if 'backbone1' in name], lr=LEARNING_RATE)
optimizer_ft_m1 = optim.SGD([param for name, param in model.named_parameters() if 'backbone1' in name], lr=LEARNING_RATE, momentum=0.9)
# adjust learning rate by gamma every step_size epochs.
lr_scheduler_m1 = torch.optim.lr_scheduler.StepLR(optimizer_ft_m1, step_size=STEP_SIZE, gamma=DECAY_RATE)
lr_scheduler_m1.step();

#optimizer_ft_m2 = optim.Adam([param for name, param in model.named_parameters() if 'backbone2' in name], lr=LEARNING_RATE)
optimizer_ft_m2 = optim.SGD([param for name, param in model.named_parameters() if 'backbone2' in name], lr=LEARNING_RATE, momentum=0.9)
# adjust learning rate by gamma every step_size epochs.
lr_scheduler_m2 = torch.optim.lr_scheduler.StepLR(optimizer_ft_m2, step_size=STEP_SIZE, gamma=DECAY_RATE)
lr_scheduler_m2.step();

optimizer_ft =  [optimizer_ft_m1, optimizer_ft_m2]
lr_scheduler =  [lr_scheduler_m1,  lr_scheduler_m2]

# define a loss function, which is a combination of sigmoid layer and the BCELoss.
# this version is more numerically stable than using plain sigmoid + BCELoss.
# the binary label distribution seems well-balanced in the dataset, otherwise, a weight-loss could be used here.
criterion = nn.BCEWithLogitsLoss(reduction='none')

train_model(model, dataloaders, criterion, optimizer_ft, lr_scheduler, device, num_epoches=TRAINING_EPOCHES)

from pickle import TUPLE
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

"""
 Filters used to pre-process raw signals.
    1. High-pass filter;
    2. Low-pass filter;
    3. Band-pass filter;
    4. Notch filter;
"""

def normalize_data(x):
   # step-1, fill nan as 0;
   #print("(max: %4f, min: %4f)" %(np.max(x), np.min(x)))
   x = np.nan_to_num(x, nan=np.mean(x))
   #print("(max: %4f, min: %4f)" %(np.max(x), np.min(x)))
   #print(np.min(x, axis=(0, 1)))
   #print(np.max(x, axis=(0, 1)))

   # ste-2: normalize data by column.
   #x_normed = (x - np.min(x, axis=(0,1), keepdims=True))/(np.max(x, axis=(0,1), keepdims=True) - np.min(x, axis=(0, 1), keepdims=True) + 0.0000001)
   x_normed = (x - np.mean(x, axis=(0,1), keepdims=True))/(np.std(x, axis=(0,1), keepdims=True) + 0.0000001)
   return x_normed


def focal_regularization(loss1, loss2):
   beta = 2.0

   # both loss1 and loss2 have the shape of [Batch, 1]
   reg = np.exp(beta * (loss1 - loss2)) - 1
   diff = loss1 - loss2
   reg[diff <= 0] = 0
   return reg


def DA_Jitter(X, sigma=0.05):
   myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
   return X + myNoise;

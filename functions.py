import torch
import torch.nn as nn                   # All neural network models, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim             # For all optimization algorithms, SGD, Adam, etc.
from torch.optim import lr_scheduler    # To change (update) the learning rate.
import torch.nn.functional as F         # All functions that don't have any parameters.
import numpy as np
import torchvision
from torchvision import datasets        # Has standard datasets that we can import in a nice way.
from torchvision import models
from torchvision import transforms      # Transformations we can perform on our datasets.
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import matplotlib
from datetime import datetime as dt
import matplotlib.gridspec as gridspec
from pandas import DataFrame
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.optim import Adam
from torch.nn import BCELoss
# torch.set_grad_enabled(True)
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from matplotlib import cm
import seaborn as sns
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score
import csv
from csv import DictWriter
import xlsxwriter
import openpyxl

# ____________________________________________PREPROCESSING FUNCTIONS___________________________________________________

def import_file(list_year, eff=''):
    df_def = pd.DataFrame()
    if isinstance(list_year, list) == True:
        for year in list_year:
            df = pd.read_csv(
                'C:/Users/ricme/Desktop/Politecnico/Tesi magistrale/TL_coding/meta_data/df_' + year + '.csv',
                encoding='latin1')
            df_def = pd.concat([df_def, df], axis=0)

    else:
        if eff != '':
            df_def = pd.read_csv(
                'C:/Users/ricme/Desktop/Politecnico/Tesi magistrale/TL_coding/meta_data/df_' + list_year + '_' + eff +'.csv',
                encoding='latin1')
        else:
            df_def = pd.read_csv('C:/Users/ricme/Desktop/Politecnico/Tesi magistrale/TL_coding/meta_data/df_' + list_year +
                             '.csv', encoding='latin1')

    del df_def['Unnamed: 0']
    del df_def['CONFROOM_BOT_1 ZN VAV TERMINAL:Zone Air Terminal VAV Damper Position[]']
    del df_def['Environment:Site Outdoor Air Relative Humidity[%]']

    return df_def


def min_max_T(df, column):
    maxT = df[column].max()
    minT = df[column].min()
    return maxT, minT


def normalization(df):
    df = (df - df.min()) / (df.max() - df.min())
    return df


# create train, val, test datasets
def create_data(df, col_name, l_train, l_val, period):
    train_mx = pd.DataFrame(df[:l_train])
    val_mx = pd.DataFrame(df[l_train:l_val])
    test_mx = pd.DataFrame(df[l_val:])
    train_mx['out'] = train_mx[col_name]
    val_mx['out'] = val_mx[col_name]
    test_mx['out'] = test_mx[col_name]
    train_mx[col_name] = train_mx[col_name].shift(periods=period) # shifting train_x
    val_mx[col_name] = val_mx[col_name].shift(periods=period)
    test_mx[col_name] = test_mx[col_name].shift(periods=period)
    train_mx = train_mx.iloc[period:] # delete the Nan
    val_mx = val_mx.iloc[period:]
    test_mx = test_mx.iloc[period:]
    train_mx = train_mx.reset_index(drop=True) # reset the index of the rows
    val_mx = val_mx.reset_index(drop=True)
    test_mx = test_mx.reset_index(drop=True)
    return train_mx, val_mx, test_mx



# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        # seq_y = sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def split_multistep_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-6:end_ix, -1]
        # seq_y = sequences[end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# _________________________________________________METRICS______________________________________________________________
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

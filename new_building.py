import torch
import torch.nn as nn                   # All neural network models, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim             # For all optimization algorithms, SGD, Adam, etc.
from torch.optim import lr_scheduler    # To change (update) the learning rate.
import torch.nn.functional as F         # All functions that don't have any parameters.
import numpy as np
from numpy import hstack
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
from functions import import_file, min_max_T, normalization, create_data, split_multistep_sequences, mean_absolute_percentage_error

# ___________________________________________IMPORT AND NORMALIZATION___________________________________________________
year='2015'
# list_3_years = ['1990', '1991', '1992']
#list_5_years = ['1990', '1991', '1992', '1993', '1994']
# list_10_years = ['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999']

df = import_file(year, eff='')
# df = import_file(list_3_years)
# df = import_file(list_5_years)
# df = import_file(list_10_years)

max_T, min_T = min_max_T(df=df, column='CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]')

# # Temperature plot
# plt.plot(df[1056:2112, -1])#'CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]'
# plt.xlim(0, 600)
# plt.title('Mean zone air temperature [??C]', size=15)
# plt.show()


df = normalization(df)

# ______________________________________________________________________________________________________________________
# 1h -> 6 timesteps
# 24h = 6*24 = 144 timesteps
# 1 weak = 144*7 = 1008 timesteps
# 1 month = 144*30 (circa) =  4320 timesteps
# ______________________________________________________________________________________________________________________

def define_period(df, time):
    if time == 'week':
        l_train = 1008+48
        l_val = int(l_train*2)
        df_def = df[:int(l_train*3)]
    if time == 'month':
        l_train = 4464
        l_val = int(l_train * 2)
        df_def = df[:int(l_train * 3)]
    if time == 'year':
        l_train = int(0.9 * len(df))  # 31536 (per un anno)
        l_val = int(l_train + 0.05 * len(df))  # da 31536 a 42048, cio?? 10512 valori (per un anno)
        df_def = df

    return df_def, l_train, l_val

# time = 'month'
df, l_train, l_val = define_period(df, time='month')


# ______________________________________Datasets_preprocessing__________________________________________________________
period = 6
# l_train = int(0.8 * len(df)) # 31536 (per un anno)
# l_val = int(l_train + 0.2*len(df)) # da 31536 a 42048, cio?? 10512 valori (per un anno)
# l_val = int(l_train+0.05*len(df)) # da 31536 a 42048, cio?? 10512 valori (per un anno)

train_df, val_df, test_df = create_data(df=df, col_name='CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]', l_train=l_train, l_val=l_val, period=period)
train_df, val_df, test_df = train_df.to_numpy(), val_df.to_numpy(), test_df.to_numpy()


# ________________________________________Splitting in X, Y data________________________________________________________
n_steps = 48 # (8 hours)
train_X, train_Y = split_multistep_sequences(train_df, n_steps)
val_X, val_Y = split_multistep_sequences(val_df, n_steps)
test_X, test_Y = split_multistep_sequences(test_df, n_steps)
#
# print(train_X.shape, train_Y.shape)
# print(val_X.shape, val_Y.shape)
# print(test_X.shape, test_Y.shape)

# Convert medium office to tensors
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).float()
val_X = torch.from_numpy(val_X).float()
val_Y = torch.from_numpy(val_Y).float()
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).float()

print(type(train_X), train_X.shape)
print(type(train_Y), train_Y.shape)
print(type(val_X), val_X.shape)
print(type(val_Y), val_Y.shape)
print(type(test_X), test_X.shape)
print(type(test_Y), test_Y.shape)


# ================================================= LSTM Structure =====================================================
# TODO: trying to use an ELU activation function

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = lookback

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        # self.af = nn.ReLU()
        # self.fc1 = nn.Linear(10, self.num_classes)

    def forward(self, x, h):
        batch_size, seq_len, _ = x.size()
        # h_0 = Variable(torch.zeros(
        #     self.num_layers, batch_size, self.hidden_size))
        #
        # c_0 = Variable(torch.zeros(
        #     self.num_layers, batch_size, self.hidden_size))
        # Propagate input through LSTM
        out, h = self.lstm(x, h)
        out = out[:, -1, :]
        #h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(out)
        # out = self.af(out)
        # out = self.fc1(out)

        return out, h

    def init_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        hidden = (hidden_state, cell_state)  # HIDDEN is defined as a TUPLE
        return hidden


# HYPER PARAMETERS
lookback = 48
lr = 0.009 #0.005 #0.009
num_layers = 5
num_hidden = 15

# generalize the number of features, timesteps and outputs
n_features = train_X.shape[2]
n_timesteps = train_X.shape[1]
n_outputs = train_Y.shape[1]

# __________________________________________________TRAINING PHASE______________________________________________________
train_batch_size =240
train_data = TensorDataset(train_X, train_Y)
train_dl = DataLoader(train_data, batch_size=train_batch_size, shuffle=False, drop_last=True)

val_batch_size = 240
val_data = TensorDataset(val_X, val_Y)
val_dl = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, drop_last=True)

# lstm = LSTM(n_features, n_timesteps)
lstm = LSTM(num_classes=n_outputs, input_size=n_features, hidden_size=num_hidden, num_layers=num_layers)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)


def train_model(model, epochs, train_dl, val_dl, optimizer, criterion, lr_scheduler, mode=''):
    # START THE TRAINING PROCESS
    model.train()
    # initialize the training loss and the validation loss
    TRAIN_LOSS = []
    VAL_LOSS = []

    for t in range(epochs):

        # TRAINING LOOP
        loss = []
        h = model.init_hidden(train_batch_size)  # hidden state is initialized at each epoch
        for x, label in train_dl:
            h = model.init_hidden(train_batch_size) # since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
            h = tuple([each.data for each in h])
            output, h = model(x.float(), h)
            #label = label.unsqueeze(1) # utilizzo .unsqueeze per non avere problemi di dimensioni
            loss_c = criterion(output, label.float())
            optimizer.zero_grad()
            loss_c.backward()
            optimizer.step()
            loss.append(loss_c.item())
        TRAIN_LOSS.append(np.sum(loss)/train_batch_size)
        if mode == 'tuning':
            lr_scheduler.step()
        # print("Epoch: %d, training loss: %1.5f" % (train_episodes, LOSS[-1]))

        # VALIDATION LOOP
        val_loss = []
        h = model.init_hidden(val_batch_size)
        for inputs, labels in val_dl:
            h = tuple([each.data for each in h])
            val_output, h = model(inputs.float(), h)
            #val_labels = labels.unsqueeze(1) # CAPIRE SE METTERLO O NO
            val_loss_c = criterion(val_output, labels.float())
            val_loss.append(val_loss_c.item())
        # VAL_LOSS.append(val_loss.item())
        VAL_LOSS.append(np.sum(val_loss)/val_batch_size)
        print('Epoch : ', t, 'Training Loss : ', TRAIN_LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])

    return TRAIN_LOSS, VAL_LOSS

epochs = 110
train_loss, val_loss = train_model(lstm, epochs=epochs, train_dl=train_dl, val_dl=val_dl, optimizer=optimizer,
                                   criterion=criterion, lr_scheduler='', mode='')


# Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(train_loss,'--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(val_loss,color='b', linewidth = 1, label = 'Validation Loss')
plt.yscale('log')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.xticks(np.arange(0, int(epochs), 10))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Multi-steps training VS Validation loss", size=15)
plt.legend()
# plt.savefig('immagini/2015/1_month/LSTM_Train_VS_Val_LOSS({}_epochs_lr_{}).png'.format(epochs, lr))
plt.show()



# ____________________________________________________SAVE THE MODEL____________________________________________________
# period = 'year'
# year = '2015'
# torch.save(lstm.state_dict(), 'train_on_' + period + '_' + str(year) + '_epochs_'+str(epochs)+'_lr_' + str(lr) + '.pth')
# torch.save(lstm.state_dict(), 'train_on_year_2015_epochs_150_lr_0.008.pth')

# Load a model
# model = LSTM(num_classes=n_outputs, input_size=n_features, hidden_size=num_hidden, num_layers=num_layers)
# period = 'week'
# year = '2015'
# model_epochs = 190
# model_lr = 0.009
# model.load_state_dict(torch.load('train_on_'+period+'_'+year+'_epochs_'+str(model_epochs)+'_lr_'+str(model_lr)+'.pth'))
# model.load_state_dict(torch.load('train_on_year_2015_epochs_100_lr_0.009.pth'))

# __________________________________________________6h PREDICTION TESTING_______________________________________________

test_batch_size = 240 # devo mettere il batch ad 1 perch?? cos?? ad ogni batch mi appendo il primo dei 6 valori predetti
test_data = TensorDataset(test_X, test_Y)
test_dl = DataLoader(test_data, shuffle=False, batch_size=test_batch_size, drop_last=True)


def test_model(model, test_dl, maxT, minT, batch_size):
    model.eval()
    h = model.init_hidden(batch_size)
    y_pred = []
    y_lab = []
    y_lab6 = []
    y_pred6 = []

    for inputs, labels in test_dl:
        h = tuple([each.data for each in h])
        test_output, h = model(inputs.float(), h)
        #labels = labels.unsqueeze(1)

        # RESCALE OUTPUTS
        test_output = test_output.detach().numpy()
        # test_output = np.reshape(test_output, (-1, 1))
        test_output = minT + test_output*(maxT-minT)

        # RESCALE LABELS
        labels = labels.detach().numpy()
        # labels = np.reshape(labels, (-1, 1))
        labels = minT + labels*(maxT-minT)

        # Append each first value of the six predicted values
                # for x in range(0, len(test_output)):
        #     if x%6 == 0:
        #         y_pred.append(test_output[x])
        #
        # for l in range(0, len(labels)):
        #     if l%6 == 0:
        #         y_lab.append(labels[l])

        y_pred.append(test_output[:, 0]) # test_output[0] per appendere solo il primo dei valori predetti ad ogni step
        y_lab.append(labels[:, 0]) # labels[0] per appendere solo il primo dei valori predetti ad ogni step
        y_pred6.append(test_output[:, 5])
        y_lab6.append(labels[:, 5])
    return y_pred, y_lab, y_pred6, y_lab6


y_pred, y_lab, y_pred6, y_lab6 = test_model(model=lstm, test_dl=test_dl, maxT=max_T, minT=min_T, batch_size=test_batch_size)


flatten = lambda l: [item for sublist in l for item in sublist]
y_pred = flatten(y_pred)
y_lab = flatten(y_lab)
y_pred = np.array(y_pred, dtype=float)
y_lab = np.array(y_lab, dtype=float)

# y_pred6 = flatten(y_pred6)
# y_lab6 = flatten(y_lab6)
# y_pred6 = np.array(y_pred6, dtype=float)
# y_lab6 = np.array(y_lab6, dtype=float)
#
# # Shift values of 6 positions because it's the sixth hour
# y_pred6 = pd.DataFrame(y_pred6)
# y_pred6 = y_pred6.shift(6, axis=0)
# y_lab6 = pd.DataFrame(y_lab6)
# y_lab6 = y_pred6.shift(6, axis=0)



error = []
error = y_pred - y_lab

plt.hist(error, 100, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-0.6, 0.6, 0.1))
plt.xlim(-0.6, 0.6)
plt.title('LSTM model 6h prediction error')
# plt.xlabel('Error')
plt.grid(True)
# plt.savefig('immagini/2015/1_year/LSTM_model_error({}_epochs_lr_{}).png'.format(epochs, lr))
plt.show()


plt.plot(y_pred, color='orange', label="Predicted")
plt.plot(y_lab, color="b", linestyle="dashed", linewidth=1, label="Real")
# plt.plot(y_pred6, color='green', label="Predicted6")
# plt.plot(y_lab6, color="b", linewidth=1, label="Real6")# , linestyle="purple"
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(0, 600)
plt.ylabel('Mean Air Temperature [??C]')
plt.xlabel('Time [h]')
plt.title("6h prediction: Real VS predicted temperature", size=15)
plt.legend()
# plt.savefig('immagini/2015/1_year/LSTM_real_VS_predicted_temperature({}_epochs_lr_{}).png'.format(epochs, lr))
plt.show()


# METRICS
MAPE = mean_absolute_percentage_error(y_lab, y_pred)
MSE = mean_squared_error(y_lab, y_pred)
R2 = r2_score(y_lab, y_pred)

print('MAPE:%0.5f%%'%MAPE)
print('MSE:', MSE.item())
print('R2:', R2.item())


plt.scatter(y_lab, y_pred,  color='k', edgecolor= 'white', linewidth=1, alpha=0.5)
plt.text(18, 24.2, 'MAPE: {:.3f}'.format(MAPE), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))
plt.text(18, 26.2, 'MSE: {:.3f}'.format(MSE), fontsize=15, bbox=dict(facecolor='green', alpha=0.5))
plt.plot([18, 28], [18, 28], color='red')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [??C]')
plt.ylabel('Predicted Temperature [??C]')
plt.title("6h prediction: Prediction distribution", size=15)
# plt.savefig('immagini/2015/1_year/LSTM_prediction_distribution({}_epochs_lr_{}).png'.format(epochs, lr))
plt.show()







# =========================================== NEW DATASET ==============================================================

df_2016 = pd.read_csv('C:/Users/ricme/Desktop/Politecnico/Tesi magistrale/TL_coding/meta_data/df_2016_high.csv', encoding='latin1')
del df_2016['Unnamed: 0']
del df_2016['CONFROOM_BOT_1 ZN VAV TERMINAL:Zone Air Terminal VAV Damper Position[]']
del df_2016['Environment:Site Outdoor Air Relative Humidity[%]']

max_T1, min_T1 = min_max_T(df_2016, 'CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]')
df_2016 = normalization(df_2016)
# 30 giorni = 4320 timesteps
# 31 giorni = 4464 timesteps
# df_2016 = df_2016[:13248] # Solo Gennaio 2016
# ______________________________________Datasets_preprocessing__________________________________________________________
period = 6
df_2016, l_train, l_val = define_period(df_2016, time='year')

# l_val = int(l_train + 0.3*len(df_2016))
# l_val = 0 # da 31536 a 42048, cio?? 10512 valori (per un anno)

train_df1, val_df1, test_df1 = create_data(df=df_2016, col_name='CONFROOM_BOT_1 ZN:Zone Mean Air Temperature[C]', l_train=l_train, l_val=l_val, period=period)
train_df1, val_df1, test_df1 = train_df1.to_numpy(), val_df1.to_numpy(), test_df1.to_numpy()


# Split medium office
n_steps1=48
train_X1, train_Y1 = split_multistep_sequences(sequences=train_df1, n_steps=n_steps)
val_X1, val_Y1 = split_multistep_sequences(sequences=val_df1, n_steps=n_steps)
test_X1, test_Y1 = split_multistep_sequences(sequences=test_df1, n_steps=n_steps1)


# Convert medium office to tensors

train_X1 = torch.from_numpy(train_X1)
train_Y1 = torch.from_numpy(train_Y1)
val_X1 = torch.from_numpy(val_X1)
val_Y1 = torch.from_numpy(val_Y1)
test_X1 = torch.from_numpy(test_X1)
test_Y1= torch.from_numpy(test_Y1)


print(type(train_X1), train_X1.shape)
print(type(train_Y1), train_Y1.shape)
print(type(val_X1), val_X1.shape)
print(type(val_Y1), val_Y1.shape)
print(type(test_X1), test_X1.shape)
print(type(test_Y1), test_Y1.shape)

#
# train_batch_size = 60
# train_data1 = TensorDataset(train_X1, train_Y1)
# train_dl1 = DataLoader(train_data1, batch_size=train_batch_size, shuffle=True, drop_last=True)
#
# val_batch_size = 60
# val_data1 = TensorDataset(val_X1, val_Y1)
# val_dl1 = DataLoader(val_data1, batch_size=val_batch_size, shuffle=True, drop_last=True)


# generalize the number of features and the number of timesteps by linking them to the preprocessing
n_features = test_X1.shape[2]
# n_timesteps = 48


# ___________________________________________________TESTING____________________________________________________________
test_batch_size1 = 400
test_data1 = TensorDataset(test_X1, test_Y1)
test_dl1 = DataLoader(test_data1, shuffle=False, batch_size=test_batch_size1, drop_last=True)
# test_losses1 = []

y_pred1, yreal1 = test_model(model, test_dl1, max_T1, min_T1, test_batch_size1)


flatten = lambda l: [item for sublist in l for item in sublist]
y_pred1 = flatten(y_pred1)
yreal1 = flatten(yreal1)
y_pred1 = np.array(y_pred1, dtype=float)
yreal1 = np.array(yreal1, dtype = float)


error1 = []
error1 = y_pred1 - yreal1

plt.hist(error1, 100, linewidth=1.5, edgecolor='black', color='orange')
plt.xticks(np.arange(-1, 0.6, 0.1))
plt.xlim(-0.6, 0.6)
plt.title('testing: model prediction error')
# plt.xlabel('Error')
plt.grid(True)
plt.savefig('immagini/2015/1_year/test_on_1_month_2016/LSTM_model_error({}_epochs_lr_{}).png'.format(epochs, lr))
plt.show()


plt.plot(y_pred1, color='orange', label="Predicted")
plt.plot(yreal1, color="b", linestyle="dashed", linewidth=1, label="Real")
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(left=0, right=600)
plt.ylabel('Mean Air Temperature [??C]')
plt.xlabel('Time [h]')
plt.title("testing: real VS predicted temperature", size=15)
plt.legend()
plt.savefig('immagini/2015/1_year/test_on_1_month_2016/LSTM_real_VS_predicted_temperature({}_epochs_lr_{}).png'.format(epochs, lr))
plt.show()


MAPE1 = mean_absolute_percentage_error(yreal1, y_pred1)
MSE1 = mean_squared_error(yreal1, y_pred1)
R21 = r2_score(yreal1, y_pred1)

print('MAPE:%0.5f%%'%MAPE1)
print('MSE:', MSE1.item())
print('R2:', R21.item())


plt.scatter(yreal1, y_pred1,  color='k', edgecolor= 'white', linewidth=1, alpha=0.8)
plt.text(18, 26.2, 'MAPE: {:.3f}'.format(MAPE1), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))
plt.text(18, 29.2, 'MSE: {:.3f}'.format(MSE1), fontsize=15, bbox=dict(facecolor='green', alpha=0.5))
plt.plot([18, 28, 30], [18, 28, 30], color='red')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlabel('Real Temperature [??C]')
plt.ylabel('Predicted Temperature [??C]')
plt.title("2001 testing: prediction distribution", size=15)
plt.savefig('immagini/2015/1_year/test_on_1_month_2016/LSTM_prediction_distribution({}_epochs_lr_{}).png'.format(epochs, lr))
plt.show()
















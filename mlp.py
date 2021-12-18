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
# plt.title('Mean zone air temperature [°C]', size=15)
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
        l_train = int(0.8 * len(df))  # 31536 (per un anno)
        l_val = int(l_train + 0.1 * len(df))  # da 31536 a 42048, cioè 10512 valori (per un anno)
        df_def = df

    return df_def, l_train, l_val

# time = 'month'
df, l_train, l_val = define_period(df, time='year')


# ______________________________________Datasets_preprocessing__________________________________________________________
period = 6
# l_train = int(0.8 * len(df)) # 31536 (per un anno)
# l_val = int(l_train + 0.2*len(df)) # da 31536 a 42048, cioè 10512 valori (per un anno)
# l_val = int(l_train+0.05*len(df)) # da 31536 a 42048, cioè 10512 valori (per un anno)

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



# ________________________________________________MLP NETWORK ___________________________________________________________
# Multivariate model definition
class MLP(nn.Module):
    # define model elements
    def __init__(self, n_features):
        super(MLP, self).__init__()
        self.hidden1 = Linear(n_features, 100) # input to first hidden layer
        self.act1 = ReLU()
        self.hidden2 = Linear(100, 100)
        self.act2 = ReLU()
        self.hidden3 = Linear(100, 100)
        self.act3 = ReLU()
        self.hidden4 = Linear(100, 100)
        self.act4 = ReLU()
        self.hidden5 = Linear(100, 100)
        self.act5 = ReLU()
        self.hidden6 = Linear(100, 6)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        # fourth hidden layer and output
        X = self.hidden4(X)
        X = self.act4(X)
        # fifth hidden layer and output
        X = self.hidden5(X)
        X = self.act5(X)
        # sexth hidden layer and output
        X = self.hidden6(X)

        return X


# __________________________________________________TRAINING PHASE______________________________________________________
train_batch_size = 500
train_data = TensorDataset(train_X, train_Y)
train_dl = DataLoader(train_data, batch_size=train_batch_size, shuffle=False, drop_last=True)

val_batch_size = 500
val_data = TensorDataset(val_X, val_Y)
val_dl = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, drop_last=True)


# PARAMETERS
lr = 0.008
n_timestep = 48
features = 8
# lstm = LSTM(n_features, n_timesteps)

# n_features = n_timestep*train_batch_size
n_features = 384 # 48 * 8
mlp = MLP(n_features)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)



def train_model(model, epochs, train_dl, val_dl, optimizer, criterion, lr_scheduler, mode=''):
    # START THE TRAINING PROCESS
    model.train()
    # initialize the training loss and the validation loss
    TRAIN_LOSS = []
    VAL_LOSS = []

    for t in range(epochs):

        # TRAINING LOOP
        loss = []
        # h = model.init_hidden(train_batch_size)  # hidden state is initialized at each epoch
        for x, label in train_dl:
            # h = model.init_hidden(train_batch_size) # since the batch is big enough, a stateless mode is used (also considering the possibility to shuffle the training examples, which increase the generalization ability of the network)
            # h = tuple([each.data for each in h])
            x = x.reshape(-1, n_features)
            # x = x.view(x.size(0), -1)
            output = model(x.float())
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
        # h = model.init_hidden(val_batch_size)
        for inputs, labels in val_dl:
            inputs = inputs.reshape(-1, n_features)
            # x = x.view(x.size(0), -1)
            # h = tuple([each.data for each in h])
            val_output = model(inputs.float())
            #val_labels = labels.unsqueeze(1) # CAPIRE SE METTERLO O NO
            val_loss_c = criterion(val_output, labels.float())
            val_loss.append(val_loss_c.item())
        # VAL_LOSS.append(val_loss.item())
        VAL_LOSS.append(np.sum(val_loss)/val_batch_size)
        print('Epoch : ', t, 'Training Loss : ', TRAIN_LOSS[-1], 'Validation Loss :', VAL_LOSS[-1])

    return TRAIN_LOSS, VAL_LOSS

#
# def train_model(model, epochs, train_dl, val_dl, optimizer, train_batch_size, val_batch_size, mode=''):
#     model.train()
#     train_loss = []
#     val_loss = []
#     # Training with multiple epochs
#     for epoch in range(epochs):
#         # ________________TRAINING_______________________________
#         total_loss = 0
#         for x, y in train_dl: # get batch
#             input = x.reshape(-1, n_features)
#             output = model(input.float())
#             loss = F.mse_loss(output.view(-1), y.float()) # calculate the loss
#
#             loss.backward() # calculate the gradient
#             optimizer.step() # update weight
#             optimizer.zero_grad()
#
#             total_loss += loss.item()
#         train_loss.append(total_loss)
#         # total_correct += get_num_correct(preds, labels)
#         if mode == 'tuning':
#             lr_scheduler.step()
#         print("epoch: ", epoch, "loss: ", total_loss/train_batch_size)
#
#         # ________________VALIDATION_____________________________
#         valid_total_loss = 0
#         for n, m in val_dl:
#             input = n.reshape(-1, n_features)
#             output = model(input.float())
#             v_loss_m = F.mse_loss(output.view(-1), m.float())  # calculate the loss
#             valid_total_loss += v_loss_m.item()
#         val_loss.append(valid_total_loss)
#         print("epoch: ", epoch, "validation loss: ", valid_total_loss/val_batch_size)
#     return train_loss, val_loss

epochs = 450
train_loss, val_loss = train_model(mlp, epochs=epochs, train_dl=train_dl, val_dl=val_dl, optimizer=optimizer, criterion=criterion, lr_scheduler='', mode='')

# train_loss, val_loss = train_model(mlp, epochs=epochs, train_dl=train_dl, val_dl=val_dl, optimizer=optimizer, train_batch_size=train_batch_size, val_batch_size=val_batch_size, mode='')



# Plot to verify validation and train loss, in order to avoid underfitting and overfitting
plt.plot(train_loss,'--',color='r', linewidth = 1, label = 'Train Loss')
plt.plot(val_loss,color='b', linewidth = 1, label = 'Validation Loss')
plt.yscale('log')
plt.ylabel('Loss (MSE)')
plt.xlabel('Epoch')
plt.xticks(np.arange(0, int(epochs), 50))
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title("Multi-steps training VS Validation loss", size=15)
plt.legend()
# plt.savefig('immagini/2015/mlp/MLP_Train_VS_Val_LOSS({}_epochs_lr_{}).png'.format(epochs, lr))
plt.show()


# ____________________________________________________SAVE THE MODEL____________________________________________________
period = 'year'
year = '2015'
# torch.save(mlp.state_dict(), 'MLP_train_on_' + period + '_' + str(year) + '_epochs_'+str(epochs)+'_lr_' + str(lr) + '.pth')
# torch.save(lstm.state_dict(), 'train_on_year_2015_epochs_150_lr_0.008.pth')

# Load a model
# model = MLP(n_features)
# period = 'week'
# year = '2015'
# model_epochs = 190
# model_lr = 0.009
# model.load_state_dict(torch.load('train_on_'+period+'_'+year+'_epochs_'+str(model_epochs)+'_lr_'+str(model_lr)+'.pth'))
# model.load_state_dict(torch.load('train_on_10_years_2015_epochs_25_lr_0.008_batch_2000.pth'))

# __________________________________________________6h PREDICTION TESTING_______________________________________________

test_batch_size = 400 # devo mettere il batch ad 1 perchè così ad ogni batch mi appendo il primo dei 6 valori predetti
test_data = TensorDataset(test_X, test_Y)
test_dl = DataLoader(test_data, shuffle=False, batch_size=test_batch_size, drop_last=True)


def test_model(model, test_dl, maxT, minT, batch_size):
    model.eval()
    # h = model.init_hidden(batch_size)
    y_pred = []
    y_lab = []
    y_lab6 = []
    y_pred6 = []

    for inputs, labels in test_dl:
        # h = tuple([each.data for each in h])
        # test_output, h = model(inputs.float(), h)
        # labels = labels.unsqueeze(1)

        inputs = inputs.reshape(-1, n_features)
        output = model(inputs.float())
        # loss = F.mse_loss(output.view(-1), labels.float())

        # RESCALE OUTPUTS
        output = output.detach().numpy()
        # # test_output = np.reshape(test_output, (-1, 1))
        test_output = minT + output*(maxT-minT)

        # RESCALE LABELS
        labels = labels.detach().numpy()
        # # labels = np.reshape(labels, (-1, 1))
        labels = minT + labels*(maxT-minT)

        y_pred.append(test_output[:, 0]) # test_output[0] per appendere solo il primo dei valori predetti ad ogni step
        y_lab.append(labels[:, 0]) # labels[0] per appendere solo il primo dei valori predetti ad ogni step
        y_pred6.append(test_output[:, 5])
        y_lab6.append(labels[:, 5])
    return y_pred, y_lab, y_pred6, y_lab6


y_pred, y_lab, y_pred6, y_lab6 = test_model(model=mlp, test_dl=test_dl, maxT=max_T, minT=min_T, batch_size=test_batch_size)


flatten = lambda l: [item for sublist in l for item in sublist]
y_pred = flatten(y_pred)
y_lab = flatten(y_lab)
y_pred = np.array(y_pred, dtype=float)
y_lab = np.array(y_lab, dtype=float)
#
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
# plt.savefig('immagini/2015/mlp/MLP_model_error({}_epochs_lr_{}_with_sixth_hour).png'.format(epochs, lr))
plt.show()


plt.plot(y_pred, color='orange', label="Predicted")
plt.plot(y_lab, color="b", linestyle="dashed", linewidth=1, label="Real")
# plt.plot(y_pred6, color='green', label="Predicted6")
# plt.plot(y_lab6, color="b", linewidth=1, label="Real6")# , linestyle="purple"
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.xlim(0, 600)
plt.ylabel('Mean Air Temperature [°C]')
plt.xlabel('Time [h]')
plt.title("6h prediction: Real VS predicted temperature", size=15)
plt.legend()
# plt.savefig('immagini/2015/mlp/MLP_real_VS_predicted_temperature({}_epochs_lr_{}_with_sixth_hour).png'.format(epochs, lr))
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
plt.xlabel('Real Temperature [°C]')
plt.ylabel('Predicted Temperature [°C]')
plt.title("6h prediction: Prediction distribution", size=15)
# plt.savefig('immagini/2015/mlp/MLP_prediction_distribution({}_epochs_lr_{}_with_sixth_hour).png'.format(epochs, lr))
plt.show()
















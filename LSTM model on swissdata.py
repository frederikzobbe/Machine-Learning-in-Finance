# Navn: LSTM model on swissdata
# Oprettet: 03-06-2022
# Senest ændret: 03-06-2022

################### CHANGELOG ###########################
# FZC: Oprettede programmet                             #              
# FZC: Tilføjede linjer til at indlæse data             #
################### DESCRIPTION #########################
# Programmet indlæser swissdata                         #
#########################################################

# 1. Reading in packages
import pandas as pd
import os as os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fix random seed for reproducibility:
np.random.seed(42)

# Convert an array of values into a dataset matrix:
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0:2]  # NUMBER OF FEATURES
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Load the dataset:
dataframe_full = pd.DataFrame(daxdatah) # loaded from 'Reading in data'
dataframe = pd.DataFrame(dataframe_full[['Close','Volume']]) # loaded from 'Reading in data'
dataset = dataframe.values
dataset = dataset.astype('float32')

# Split into train and test sets
perctest = 0.7
train_size = int(len(dataset) * perctest)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# MISSING: Should scale inputs (features) and outputs (prices) separately
# Normalize the dataset (*after splitting)
scaler = MinMaxScaler(feature_range=(0, 1))
train  = scaler.fit_transform(train)
test   = scaler.transform(test)

# Reshape into X = t and Y = t + 1
look_back = 20
trainX, trainY = create_dataset(train, look_back)
testX, testY   = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 2))  # NUMBER OF FEATURES
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 2))  # NUMBER OF FEATURES

trainX = torch.tensor(trainX, dtype=torch.float)
trainY = torch.tensor(trainY, dtype=torch.float)
testX = torch.tensor(testX, dtype=torch.float)
testY = torch.tensor(testY, dtype=torch.float)

class Net(nn.Module):
    def __init__(self, hidden_size=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)  # NUMBER OF FEATURES
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x, _ = self.lstm(x)  # Run LSTM and store the hidden layer outputs
        x = x[:, -1, :]      # take the last hidden layer
        x = self.linear(x)   # a normal dense layer
        return x

net = Net()

opt = torch.optim.Adam(net.parameters(), lr=0.005)
progress_bar = tqdm(range(100))
for epoch in progress_bar:
    prediction = net(trainX)
    loss = torch.sum((prediction.flatten() - trainY.flatten())**2)
    progress_bar.set_description(f'Loss = {float(loss)}')
    loss.backward()
    opt.step()
    opt.zero_grad()

# make predictions 
with torch.no_grad():
    trainPredict = net(trainX).numpy()
    testPredict = net(testX).numpy()

# make predictions (recursively)
with torch.no_grad():
    pred_recursive_test = trainX[-1,:,:].numpy()
    for i in np.arange(testX.shape[0]):
        pred_recursive_test = np.reshape(pred_recursive_test, (1, look_back+i, 1))
        input = pred_recursive_test[:,-look_back:,:]
        input_torch = torch.tensor(input, dtype=torch.float)
        tmp_out = net(input_torch).numpy()
        pred_recursive_test     = np.append(pred_recursive_test, tmp_out)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inv = scaler.inverse_transform([trainY.numpy()])
testPredict = scaler.inverse_transform(testPredict)
testY_inv = scaler.inverse_transform([testY.numpy()])

pred_recursive_test_inv    = scaler.inverse_transform([pred_recursive_test[look_back-1:]])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY_inv[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

testPredictPlot[len(trainPredict)-2+look_back:len(dataset)-3-look_back, :] = np.transpose(pred_recursive_test_inv)

# plot baseline and predictions
splitpoint = int((perctest * len(dataset)))-3
splitdate = dataframe_full[dataframe_full.index == splitpoint]['CET'].dt.date.item().strftime('%d %b %Y')

#xi = list(range(len(dataset)))
#xdates = dataframe_full[dataframe_full.index == xi]['CET'].dt.date

# Long period
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
plt.plot(dataset, label='Observations')
plt.plot(trainPredictPlot, label='Predict: Train')
plt.plot(testPredictPlot, label='Predict: Test')
ax.legend(loc='upper left', frameon=False)
plt.title('DAX index predictions')
#plt.xticks(xi, xdates)
plt.axvline(x = splitpoint, color = 'r', linestyle = '-')
plt.text(splitpoint+200,8000, str(splitdate),rotation=0)
plt.show()

# Short period (After traindata ends)
fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
plt.plot(dataset, label='Observations', color = 'b')
#plt.plot(trainPredictPlot, label='Predict: Train')
plt.plot(testPredictPlot, label='Predict: Test', color = 'r')
ax.legend(loc='upper left', frameon=False)
plt.title('DAX index predictions')
#plt.xticks(xi, xdates)
plt.axvline(x = splitpoint, color = 'r', linestyle = '-')
plt.text(splitpoint+200,8000, str(splitdate),rotation=0)
plt.xlim([splitpoint-20, splitpoint+50])
plt.ylim([12000, 15000])
plt.show()

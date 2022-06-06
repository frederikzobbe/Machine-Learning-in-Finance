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
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Load the dataset:
dataframe_full = pd.DataFrame(daxdatah) # loaded from 'Reading in data'
dataframe = pd.DataFrame(dataframe_full['Close']) # loaded from 'Reading in data'
dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets
perctest = 0.7
train_size = int(len(dataset) * perctest)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Reshape into X = t and Y = t + 1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY   = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

trainX = torch.tensor(trainX, dtype=torch.float)
trainY = torch.tensor(trainY, dtype=torch.float)
testX = torch.tensor(testX, dtype=torch.float)
testY = torch.tensor(testY, dtype=torch.float)

class Net(nn.Module):
    def __init__(self, hidden_size=7):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x, _ = self.lstm(x)  # Run LSTM and store the hidden layer outputs
        x = x[:, -1, :]      # take the last hidden layer
        x = self.linear(x)   # a normal dense layer
        return x

net = Net()

opt = torch.optim.Adam(net.parameters(), lr=0.005)
progress_bar = tqdm(range(2000))
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

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY_inv = scaler.inverse_transform([trainY.numpy()])
testPredict = scaler.inverse_transform(testPredict)
testY_inv = scaler.inverse_transform([testY.numpy()])

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
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
splitpoint = int((perctest * len(dataset)))
splitdate = dataframe_full[dataframe_full.index == splitpoint]['CET'].dt.date.item().strftime('%d %b %Y')

#xi = list(range(len(dataset)))
#xdates = dataframe_full[dataframe_full.index == xi]['CET'].dt.date

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
plt.plot(scaler.inverse_transform(dataset), label='Observations')
plt.plot(trainPredictPlot, label='Predict: Train')
plt.plot(testPredictPlot, label='Predict: Test')
ax.legend(loc='upper left', frameon=False)
plt.title('DAX index predictions')
#plt.xticks(xi, xdates)
plt.axvline(x = splitpoint, color = 'r', linestyle = '-')
plt.text(splitpoint+200,8000, str(splitdate),rotation=0)
plt.show()

int((perctest * len(dataset)))

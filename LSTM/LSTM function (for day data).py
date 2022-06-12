# Navn: LSTM function (for day data)

################### DESCRIPTION #############################
# Programmet prædikterer kursen 1 dag frem med rekalibrering
# fra en LSTM model 
#############################################################

# 1. Reading in packages
import pandas as pd
import numpy as np
import torch
import time as time
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import lightgbm as lgb

# 2. Reading in data

#os.chdir("/Users/frederikzobbe/Desktop/Data")
#df6 = pd.read_csv('GAS.CMDUSD_Candlestick_1_D_BID_01.01.2014-30.04.2022.csv')
#df6['Name'] = "GAS"
#df6['Type'] = "Commodity"
#data = timefunc(df6, 0)
#data.to_csv('DaxDataDay.txt', index = False, header=True)

# Convert an array of values into a dataset matrix:
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Fix random seed for reproducibility:
np.random.seed(42)
look_back = 30

# Load the dataset:
dataframe = data['Close']
dataset = dataframe.values
dataset = dataset.astype('float32').reshape(-1,1)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets
train_to_date = '2021-01-03'
train_to_idx = data.index[(data['Year'] == int(train_to_date[0:4])) &
                          (data['Month'] == int(train_to_date[5:7])) &
                          (data['Day'] == int(train_to_date[-2:]))][0]

train_size = train_to_idx
train, test = dataset[0:train_to_idx, :], dataset[train_to_idx-look_back-1:len(dataset),:]

#train_size = int(len(dataset) * 0.7)
#test_size = len(dataset) - train_size
#train, test = dataset[0:len(train),:], dataset[len(train)-look_back-1:len(dataset),:]

# Reshape into X = t and Y = t + 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Move to PyTorch tensors:
trainX = torch.tensor(trainX, dtype=torch.float)
trainY = torch.tensor(trainY, dtype=torch.float)
testX = torch.tensor(testX, dtype=torch.float)
testY = torch.tensor(testY, dtype=torch.float)

class Net(nn.Module):
    def __init__(self, hidden_size=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x, _ = self.lstm(x)  # Run LSTM and store the hidden layer outputs
        x = x[:, -1, :]      # take the last hidden layer
        x = self.linear(x)   # a normal dense layer
        return x

net = Net()

opt = torch.optim.Adam(net.parameters(), lr=5e-3)
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
testPredictPlot[len(trainPredict)+(look_back)+1:len(dataset), :] = testPredict

# plot baseline and predictions
fig, ax = plt.subplots(figsize=(8, 5), dpi = 130)
plt.plot(scaler.inverse_transform(dataset), label='Observations', color = 'b')
plt.plot(trainPredictPlot, label='Predict: Train', color = 'r')
plt.plot(testPredictPlot, label='Predict: Test', color = 'y', marker='.', markersize = 5, mfc='red')
ax.legend(loc='upper left', frameon=False)
#plt.xlim([len(trainPredict)+(look_back*2)+1, 1600])
plt.xlim([train_to_idx-50, len(data)])
#plt.ylim([12500,17000])
plt.show()

# Saving the predictions 
data_sub = data.iloc[:,[3,5,6,7]]
data_sub.iloc[train_size:,:]
data_sub['Predictions'] = np.nan
data_sub.iloc[train_size:,4] = testPredict



data_sub.iloc[train_to_idx-1:,:]


#daxdata = data_sub
#nasdata = data_sub
#gasdata = data_sub

df = pd.concat([daxdata, nasdata, gasdata])
df.reset_index(drop=True, inplace=True)
df.to_csv('LasseDataDay.txt', index = False, header=True)

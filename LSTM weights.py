# 1. Reading in packages
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import os as os

device = "cuda" if torch.cuda.is_available() else "cpu"


### --------------------------- Michael LSTM index weights ----------------------------------

IndexData20Min_DB = IndexDat20Min_DB # From "Reading in the data"
IndexData20Min_varying_DB = IndexDat20Min_varying_DB # From "reading in the data"

# Want to use indices that are not too correlated so I will try to plot them to see which are uncorrelated

# Get which indices we have data for
names = pd.unique(IndexData20Min_DB.Name)

# Indices for the different stock indexes
daxIdx = IndexData20Min_DB.Name=="DAX"
FTSEIdx = IndexData20Min_DB.Name=="FTSE"
SogPIdx = IndexData20Min_DB.Name=="S&P"
NASDAQIdx = IndexData20Min_DB.Name=="NASDAQ"

# Create dataset for each index as python is shitty and has no ggplot so cannot work with data on long format
daxdat = IndexData20Min_DB[daxIdx].loc[:,['Close', 'CET']].reset_index(drop=True)
FTSEdat = IndexData20Min_DB[FTSEIdx].loc[:,['Close', 'CET']].reset_index(drop=True)
SogPdat = IndexData20Min_DB[SogPIdx].loc[:,['Close', 'CET']].reset_index(drop=True)
NASDAQdat = IndexData20Min_DB[NASDAQIdx].loc[:,['Close', 'CET']].reset_index(drop=True)

# Want to see how they develop relative to themselves.
daxdat["ClosePct"] = daxdat.Close/daxdat.Close[0]
FTSEdat["ClosePct"] = FTSEdat.Close/FTSEdat.Close[0]
SogPdat["ClosePct"] = SogPdat.Close/SogPdat.Close[0]
NASDAQdat["ClosePct"] = NASDAQdat.Close/NASDAQdat.Close[0]


fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
plt.plot(daxdat.loc[:,'CET'], daxdat.loc[:,'ClosePct'], label='Dax')
plt.plot(FTSEdat.loc[:,'CET'], FTSEdat.loc[:,'ClosePct'], label='FTSE')
plt.plot(SogPdat.loc[:,'CET'], SogPdat.loc[:,'ClosePct'], label='S&P')
plt.plot(NASDAQdat.loc[:,'CET'], NASDAQdat.loc[:,'ClosePct'], label='NASDAQ')
ax.legend(loc='upper left', frameon=False)
plt.close()

# plotting histogram of log returns for DAX
log_return_dax = np.log(daxdat.Close/daxdat.Close.shift(1)).dropna()
bin_len = 0.001
plt.hist(log_return_dax, bins=np.arange(min(log_return_dax), max(log_return_dax)+bin_len, bin_len))
plt.close()
# All indices are VERY correlated. So lets just make a LSTM with all of them

# Data prep
# Load the dataset:
dataframe_full = pd.DataFrame(IndexData20Min_DB)  # loaded from 'Reading in data'
dataframe_full2 = pd.DataFrame(IndexData20Min_varying_DB) # loaded from 'Reading in data'

# Adding log returns to data
dataframe_full['logreturn'] = np.log(dataframe_full.Close/dataframe_full.Close.shift(1)).fillna(0)
dataframe_full2['logreturn'] = np.log(dataframe_full2.Close/dataframe_full2.Close.shift(1)).fillna(0)

# Need to have data on wide format for the different index, in order to do so we need an index
id = 0
Idx = []
for name in dataframe_full2.Name.unique():
    DAXidx = dataframe_full2.Name == name
    id = 0
    for i in np.arange(len(DAXidx)):
        if DAXidx[i]:
            Idx.append(id)
            id += 1

dataframe_full2['idx'] = Idx

# Now convert data to wide format
#dataframe_full = dataframe_full.pivot(columns='Name', index='idx')
dataframe_full2 = dataframe_full2.pivot(columns='Name', index='idx')


# Function to get training data.
def lookbackdat(dataframe,
                assetlist, # list [] of asset names
                train_to_date # format: 'YYYY-MM'
                ):
    idx = []
    for name in assetlist:
        id = dataframe.index[(dataframe[('Month', name)] == int(train_to_date[5:7])) &
                             (dataframe[('Year', name)] == int(train_to_date[0:4]))][0]
        idx.append(id)

    NoOfDB = min(idx)

    TrainDat = pd.DataFrame()
    for name, i in zip(assetlist, np.arange(len(assetlist))):
        Dat = dataframe.loc[(idx[i]-NoOfDB):(idx[i]), (slice(None), assetlist[i])].reset_index(drop=True)
        TrainDat = pd.concat([TrainDat, Dat], axis=1)

    return TrainDat

# Creating training data
assetlist = ["DAX", 'FTSE', 'NASDAQ', 'S&P'] # Fuck HK. Shitty distribution of how much money is transacted each observation
train_to_date = "2020-01"
TrainDat_varying = lookbackdat(dataframe_full2, assetlist, train_to_date)

# -------------------------- Training the model ------------------------
look_back = 20
features_used_init = ['Close', 'Volume', 'logreturn']
features_used = np.repeat(features_used_init,len(assetlist))
size_hidden = len(features_used)*3
rebalancebars = 10
learning_rate = 0.1
num_epochs = 50

# Create dataset fct
def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0:len(features_used)]  # NUMBER OF FEATURES
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0:4])
    return np.array(dataX), np.array(dataY)

dataset = TrainDat_varying
dataset.reset_index(drop=True, inplace=True)
dataframe_full = pd.DataFrame(dataset)
dataframe = pd.DataFrame(dataframe_full.loc[:, (['Close', 'Volume', 'logreturn'], slice(None))].reset_index(drop=True))
dataset_used = dataframe.values
dataset_used = dataset_used.astype('float32')

# Get train data. We want to train till the last 500 dollarbars
train_to_idx = len(dataset_used) - 500

train, test = dataset_used[0:train_to_idx, :], dataset_used[train_to_idx:len(dataset_used), :]

# scale dataset (*after splitting)
scaler_out, scaler_feat = MinMaxScaler(feature_range=(0,1)), MinMaxScaler(feature_range=(0,1))
train_price_scaled = scaler_out.fit_transform(train[:, :4])
train_feat_scaled = scaler_feat.fit_transform(train[:, 4:])
test_price_scaled = scaler_out.transform(test[:, :4])
test_feat_scaled = scaler_feat.transform(test[:, 4:])
train = np.column_stack((train_price_scaled, train_feat_scaled))
test = np.column_stack((test_price_scaled, test_feat_scaled))

# Reshape into X = t and Y = t + 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], len(features_used)))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], len(features_used)))

trainX = torch.tensor(trainX, dtype=torch.float)
trainY = torch.tensor(trainY, dtype=torch.float)
testX = torch.tensor(testX, dtype=torch.float)
testY = torch.tensor(testY, dtype=torch.float)


class Net(nn.Module):
    def __init__(self, hidden_size=size_hidden):
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(features_used), hidden_size=hidden_size,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, len(assetlist))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)  # Run LSTM and store the hidden layer outputs
        x = x[:, -1, :]  # take the last hidden layer
        x = self.linear(x)  # a normal dense layer
        x = self.softmax(x)
        return x

net = Net()

# loss_fn... Right now we use last 20 dollarbars to predict a vector of weights that sum to one...
# Everytime we get a new dollarbar we predict a new vector of weights
# Loss function could be:
# Use weight to purchase the different assets. Look 10 dollarbars forward in time and calculate negative return.
# 100000*weights = vec --> vec[0]/pris[0] etc = vec2 --> (price_rebalance[0] - price_now[0])*vec2[0]


hmm = net(trainX)
hmmtoMoney = (hmm[0:2] * 100000)
hmmtoMoney2 = [[25437.3789, 23395.6270, 26312.9473, 24854.0508],
               [25404.4629, 23482.7461, 26204.8164, 24907.9785]]
hmm2 = scaler_out.inverse_transform(trainY.numpy())
hmm3 = torch.tensor(hmm2)
hmm8 = hmmtoMoney/hmm3[0:2]

hmm4 = hmm3.roll(-(len(assetlist)*rebalancebars))
hmm5 = hmm3
hmm5[-rebalancebars:len(hmm3)] = hmm4[-rebalancebars:len(hmm4)].clone()
hmm6 = hmm5 - hmm4
hmm7 = hmm6.numpy()

hmm9 = hmm8 * hmm6[0:2]
hmm10 = torch.sum(-hmm9, axis=1)
hmm11 = torch.sum(hmm10)


opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
progress_bar = tqdm(range(num_epochs))

for epoch in progress_bar:
    # Initial financial stuff
    Capital = 100000
    prediction = net(trainX)
    MoneyAllocation = prediction * Capital
    CostOfAsset = scaler_out.inverse_transform(trainY.numpy())
    CostOfAsset = torch.tensor(CostOfAsset)
    NoOfStocks = MoneyAllocation.clone()/CostOfAsset.clone()
    CostOfAssetShifted = CostOfAsset.roll(-(len(assetlist)*rebalancebars)).clone() # Shift cost of asset array s.t. return can be calculated
    # Replace last rebalancebars observations from CostofAsset. As return from these cannot be calculated and thus should not contribute to loss
    CostOfAsset[-rebalancebars:len(CostOfAsset)] = CostOfAssetShifted[-rebalancebars:len(CostOfAssetShifted)].clone()
    Return1 = CostOfAsset - CostOfAssetShifted
    Return2 = NoOfStocks*Return1
    Return3 = torch.sum(-Return2, axis=1)
    Return4 = torch.sum(Return3)
    # Loss calculation
    loss = Return4

    progress_bar.set_description(f'Loss = {float(loss)}')
    loss.backward()
    opt.step()
    opt.zero_grad()

# make predictions
with torch.no_grad():
    trainPredict = net(trainX).numpy()
    testPredict = net(testX).numpy()

# invert Prices
trainY_inv = scaler_out.inverse_transform(trainY.numpy())

testY_inv = scaler_out.inverse_transform(testY.numpy())

def losscalc(Predict, Y_inv):
    Capital = 100000
    prediction = Predict
    MoneyAllocation = prediction * Capital
    CostOfAsset = Y_inv
    CostOfAsset = torch.tensor(CostOfAsset)
    NoOfStocks = MoneyAllocation / CostOfAsset.clone()
    CostOfAssetShifted = CostOfAsset.roll(-(len(Stocks) * rebalancebars)).clone()  # Shift cost of asset array s.t. return can be calculated
    # Replace last rebalancebars observations from CostofAsset. As return from these cannot be calculated and thus should not contribute to loss
    CostOfAsset[-rebalancebars:len(CostOfAsset)] = CostOfAssetShifted[-rebalancebars:len(CostOfAssetShifted)].clone()
    Return1 = CostOfAsset - CostOfAssetShifted
    Return2 = NoOfStocks * Return1
    Return3 = torch.sum(Return2, axis=1)
    Return4 = torch.sum(Return3)
    # Loss calculation
    loss = Return4

    return loss

# calculate root mean squared error
trainScore = losscalc(trainPredict, trainY_inv)
print('Train Score: %.2f Negative Return' % (trainScore))
testScore = losscalc(testPredict, testY_inv)
print('Test Score: %.2f Negative Return' % (testScore))

# Test loss if we just bought the other indices...
ManualPredict = np.repeat([[0,0,0,1],[0,0,0,1]], [3532,1], axis=0)
losscalc(ManualPredict, trainY_inv) # Loss would be positive for all other idx. So we would have lost money... what the frick

# Algorithm just wants to hold FTSE lol...
# Lets plot dollarbar returns to see why
DAXdat = dataframe.loc[:, (slice(None), 'DAX')].droplevel('Name',axis=1)
FTSEDAT = dataframe.loc[:, (slice(None), 'FTSE')].droplevel('Name', axis=1)
NASDAQdat = dataframe.loc[:, (slice(None), 'NASDAQ')].droplevel('Name', axis=1)
SPdat = dataframe.loc[:, (slice(None), 'S&P')].droplevel('Name', axis=1)


fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
plt.plot(DAXdat.loc[:,'Close'], label='Dax')
plt.plot(FTSEDAT.loc[:,'Close'], label='FTSE')
plt.plot(NASDAQdat.loc[:,'Close'], label='NASDAQ')
plt.plot(SPdat.loc[:,'Close'], label='S&P')
ax.legend(loc='upper left', frameon=False)
plt.title("Price development for indexes on dollar bar level")
plt.xlabel("DollarBar")
plt.ylabel('Price')
plt.close()

# Looks like NASDAQ would have been better. So lets try and reset weights and train model again

for name, module in net.named_children():
    if hasattr(module, 'reset_parameters'):
        print('resetting ', name)
        module.reset_parameters()



## ------------------ Portfolio optimization timedata ----------------------------------

IndexDat = PortfolioDat

# Calculating log returns
IndexDat['logreturn'] = np.log(IndexDat.Close/IndexDat.Close.shift(1)).fillna(0)

# Need to have data on wide format for the different index, in order to do so we need an index
id = 0
Idx = []

for name in IndexDat.Name.unique():
    DAXidx = IndexDat.Name == name
    DAXidx = np.array(DAXidx)
    id = 0
    for i in np.arange(len(DAXidx)):
        if DAXidx[i]:
            Idx.append(id)
            id += 1

IndexDat['idx'] = Idx
IndexDatWide = IndexDat.pivot(columns='Name', index='idx')

# Wish to remove rows where volume for all indices are 0
VolumeHelper = IndexDatWide.loc[:, (['Volume'], slice(None))]
VolumeIdx = VolumeHelper.sum(axis=1)!=0

IndexDatWide = IndexDatWide[VolumeIdx]
IndexDatWide.columns
# ----------------------- LSTM implementation -----------------------------------
look_back = 50
features_used_init = ['Close', 'Volume', 'logreturn']
Stocks = ['DAX', 'FTSE', 'HK', 'NASDAQ', 'S&P', 'Coffee', 'GAS']
noStocks = len(Stocks)
features_used = np.repeat(features_used_init,len(Stocks))
size_hidden = len(features_used)*3
rebalancebars = 24
learning_rate = 0.01
num_epochs = 10

# Create dataset fct. Remember to update which columns to append to dataY.
def create_dataset(dataset, look_back=20):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0:len(features_used)]  # NUMBER OF FEATURES
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0:noStocks])
    return np.array(dataX), np.array(dataY)

dataset = IndexDatWide
dataset.reset_index(drop=True, inplace=True)
dataframe_full = pd.DataFrame(dataset)
dataframe = pd.DataFrame(dataframe_full.loc[:, (['Close', 'Volume', 'logreturn'], slice(None))].reset_index(drop=True))
dataset_used = dataframe.values
dataset_used = dataset_used.astype('float32')

# Get train data. We want to train till the last 500 dollarbars
train_to_idx = len(dataset_used) - 20000

train, test = dataset_used[0:train_to_idx, :], dataset_used[train_to_idx:(len(dataset_used)-10000), :]
val = dataset_used[(len(dataset_used)-10000):len(dataset_used),:]

# scale dataset (*after splitting)
scaler_out, scaler_feat = MinMaxScaler(feature_range=(0,1)), MinMaxScaler(feature_range=(0,1))
train_price_scaled = scaler_out.fit_transform(train[:, :noStocks])
train_feat_scaled = scaler_feat.fit_transform(train[:, noStocks:])
test_price_scaled = scaler_out.transform(test[:, :noStocks])
test_feat_scaled = scaler_feat.transform(test[:, noStocks:])
val_price_scaled = scaler_out.transform(val[:, :noStocks])
val_feat_scaled = scaler_feat.transform(val[:,noStocks:])
train = np.column_stack((train_price_scaled, train_feat_scaled))
test = np.column_stack((test_price_scaled, test_feat_scaled))
val = np.column_stack((val_price_scaled, val_feat_scaled))

# Reshape into X = t and Y = t + 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
valX, valY = create_dataset(val, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], len(features_used)))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], len(features_used)))
valX = np.reshape(testX, (valX.shape[0], valX.shape[1], len(features_used)))

trainX = torch.tensor(trainX, dtype=torch.float)
trainY = torch.tensor(trainY, dtype=torch.float)
testX = torch.tensor(testX, dtype=torch.float)
testY = torch.tensor(testY, dtype=torch.float)
valX = torch.tensor(valX, dtype=torch.float)
valY = torch.tensor(valY, dtype=torch.float)

class Net(nn.Module):
    def __init__(self, hidden_size=size_hidden):
        super().__init__()
        self.lstm = nn.LSTM(input_size=len(features_used), hidden_size=hidden_size,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, len(Stocks))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm(x)  # Run LSTM and store the hidden layer outputs
        x = x[:, -1, :]  # take the last hidden layer
        x = self.linear(x)  # a normal dense layer
        x = self.softmax(x)
        return x

net = Net()

# loss_fn... Right now we use last 20 dollarbars to predict a vector of weights that sum to one...
# Everytime we get a new dollarbar we predict a new vector of weights
# Loss function could be:
# Use weight to purchase the different assets. Look 10 dollarbars forward in time and calculate negative return.
# 100000*weights = vec --> vec[0]/pris[0] etc = vec2 --> (price_rebalance[0] - price_now[0])*vec2[0]

val_scores = []
train_scores = []

opt = torch.optim.SGD(net.parameters(), lr=learning_rate)
progress_bar = tqdm(range(num_epochs))

for epoch in progress_bar:
    # Initial financial stuff
    Capital = 100000
    prediction = net(trainX)
    MoneyAllocation = prediction * Capital
    CostOfAsset = scaler_out.inverse_transform(trainY.numpy())
    CostOfAsset = torch.tensor(CostOfAsset)
    NoOfStocks = MoneyAllocation.clone()/CostOfAsset.clone()
    CostOfAssetShifted = CostOfAsset.roll(-(len(Stocks)*rebalancebars)).clone() # Shift cost of asset array s.t. return can be calculated
    # Replace last rebalancebars observations from CostofAsset. As return from these cannot be calculated and thus should not contribute to loss
    CostOfAsset[-rebalancebars:len(CostOfAsset)] = CostOfAssetShifted[-rebalancebars:len(CostOfAssetShifted)].clone()
    Return1 = CostOfAssetShifted - CostOfAsset
    Return2 = NoOfStocks*Return1
    Return3 = torch.sum(-Return2, axis=1)
    Return4 = torch.sum(Return3)

    # Loss calculation
    loss = Return4

    progress_bar.set_description(f'Loss = {float(loss)}')
    loss.backward()
    opt.step()
    opt.zero_grad()

# Diversification pen
    _, pen = torch.max(prediction.clone(), axis=1)
+ sum(pen*10)

# Calculate test loss
    # Initial financial stuff
    Capital = 100000
    with torch.no_grad():
        testprediction = net(testX)
        MoneyAllocation_test = testprediction * Capital
        CostOfAsset_test = scaler_out.inverse_transform(testY.numpy())
        CostOfAsset_test = torch.tensor(CostOfAsset_test)
        NoOfStocks_test = MoneyAllocation_test.clone() / CostOfAsset_test.clone()
        CostOfAssetShifted_test = CostOfAsset_test.roll(-(len(Stocks) * rebalancebars)).clone()  # Shift cost of asset array s.t. return can be calculated
        # Replace last rebalancebars observations from CostofAsset. As return from these cannot be calculated and thus should not contribute to loss
        CostOfAsset_test[-rebalancebars:len(CostOfAsset_test)] = CostOfAssetShifted_test[-rebalancebars:len(CostOfAssetShifted_test)].clone()
        Return1_test = CostOfAssetShifted_test - CostOfAsset_test
        Return2_test = NoOfStocks_test * Return1_test
        Return3_test = torch.sum(-Return2_test, axis=1)
        Return4_test = torch.sum(Return3_test)
        # Loss calculation
        loss_test = Return4_test

        # Save test and train loss
        train_score = loss.numpy()
        val_score = loss_test.numpy()
        train_scores.append(train_score)
        val_scores.append(val_score)


# Plot test and validation results from the model
fig, ax = plt.subplots(figsize=(10, 6), dpi = 100) # dpi = 500 for saves
X = np.arange(len(val_scores))
y = val_scores
z = train_scores
# Plotting both the curves simultaneously
plt.plot(X, y, color='r', label='Validation loss')
plt.plot(X, z, color='g', label='Train loss')
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and validation loss")
plt.legend()
plt.close()
#plt.savefig('Trainloss.png')

# Huge overtraining it seems like... What is validation loss?

calcpred = net(valX)
valY_inv = scaler_out.inverse_transform(valY.numpy())
losscalc(calcpred, valY_inv)


# Reset weights
for name, module in net.named_children():
    if hasattr(module, 'reset_parameters'):
        print('resetting ', name)
        module.reset_parameters()

# plotting returns of the different indices
DAXdat = IndexDatWide.loc[:, (slice(None), 'DAX')].droplevel('Name',axis=1)
FTSEDAT = IndexDatWide.loc[:, (slice(None), 'FTSE')].droplevel('Name', axis=1)
NASDAQdat = IndexDatWide.loc[:, (slice(None), 'NASDAQ')].droplevel('Name', axis=1)
SPdat = IndexDatWide.loc[:, (slice(None), 'S&P')].droplevel('Name', axis=1)
HKdat = IndexDatWide.loc[:, (slice(None), 'HK')].droplevel('Name', axis=1)
CoffeeDat = IndexDatWide.loc[:, (slice(None), 'Coffee')].droplevel('Name', axis=1)
GASdat = IndexDatWide.loc[:, (slice(None), 'GAS')].droplevel('Name', axis=1)

# Want to see how they develop relative to themselves.
DAXdat["ClosePct"] = DAXdat.Close/DAXdat.Close[0]
FTSEDAT["ClosePct"] = FTSEDAT.Close/FTSEDAT.Close[0]
SPdat["ClosePct"] = SPdat.Close/SPdat.Close[0]
NASDAQdat["ClosePct"] = NASDAQdat.Close/NASDAQdat.Close[0]
HKdat['ClosePct'] = HKdat.Close/HKdat.Close[0]
CoffeeDat['ClosePct'] = CoffeeDat.Close/CoffeeDat.Close[0]
GASdat['ClosePct'] = GASdat.Close/GASdat.Close[0]

fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
plt.plot(DAXdat.loc[:,'CET'], DAXdat.loc[:,'ClosePct'], label='Dax')
plt.plot(FTSEDAT.loc[:,'CET'], FTSEDAT.loc[:,'ClosePct'], label='FTSE')
plt.plot(NASDAQdat.loc[:,'CET'], NASDAQdat.loc[:,'ClosePct'], label='NASDAQ')
plt.plot(SPdat.loc[:,'CET'], SPdat.loc[:,'ClosePct'], label='S&P')
plt.plot(HKdat.loc[:,'CET'], HKdat.loc[:,'ClosePct'], label='HK')
plt.plot(CoffeeDat.loc[:,'CET'], CoffeeDat.loc[:,'ClosePct'], label='Coffee')
plt.plot(GASdat.loc[:,'CET'], GASdat.loc[:,'ClosePct'], label='GAS')
ax.legend(loc='upper left', frameon=False)
plt.title("Relative price development")
plt.xlabel("Date")
plt.axvline(x=CoffeeDat.CET[len(DAXdat)-20000], color='black')
plt.ylabel('Price')
plt.close()

# Check that just investing in FTSE really is the best idea
trainY_inv = scaler_out.inverse_transform(trainY.numpy())
ManualPredict = np.repeat([[0,0,0,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,0,0,1]], [14882,977,41054], axis=0)
losscalc(ManualPredict, trainY_inv) # Loss would be positive for all other idx. So we would have lost money... what the frick
ManualPredict2 = np.repeat([[0,0,0,0,0,0,1],[0,0,0,0,0,0,1]], [56912,1], axis=0)
losscalc(ManualPredict2, trainY_inv)

56913-39137
15859 - 14882
56913 - 15859

GasClose = GASdat.Close
hmm = GasClose[-24:len(GasClose)].reindex(np.arange(76940, 76964,1))
hmm[:] = GasClose[-24:len(GasClose)]
GasClose_Shifted = GasClose.shift(-24).fillna(hmm)
ShiftedReturns = GasClose_Shifted - GasClose

GASdat['return'] = ShiftedReturns

hmm2 = pd.Series(np.arange(76944, 76964,1))
test = dataframe.values
test = test.astype('float32')
test_roll = np.roll(test, -(len(Stocks) * rebalancebars))
k = np.arange(10)
np.roll(test, 10)
## -------------------------- Notes --------------------------------------------
# Close price is open price for current index + 1


# Want to figure out why HK has so few dollar bars. Turns out HK idx has ridicolous variance in data.
IndexDat20Min = IndexDat20Min # From Reading in the data

IndexDat20MinHK = IndexDat20Min[IndexDat20Min.Name == "HK"]
IndexDat20MinHK['Price'] = IndexDat20MinHK.Volume * (IndexDat20MinHK.Open + IndexDat20MinHK.Close)/2
IndexDat20MinHK = IndexDat20MinHK[IndexDat20MinHK.Price > 0]
plt.hist(IndexDat20MinHK.Price, bins=np.arange(min(IndexDat20MinHK.Price),max(IndexDat20MinHK.Price), 8e+9))
IndexDat20MinHK.Price.describe()
plt.close()
hmm = IndexDat20MinHK[IndexDat20MinHK.Price > 2.500000e+10]
np.median(IndexDat20MinHK.Price)
np.mean(IndexDat20MinHK.Price)

# How to get sub data
hmm = dataframe_full2.loc[0:10,(slice(None), 'DAX')]
hmm2 = dataframe_full2.loc[0:10,(slice(None), 'FTSE')]

hmm3 = pd.concat([hmm, hmm2], axis=1)

# Working with multilevel
features_used = ['Close', 'Volume', 'logreturn']

hmm = TrainDat_varying.loc[:, (['Close', 'Volume', 'logreturn'], slice(None))].reset_index(drop=True)
test = hmm.droplevel()
hmm.MultiIndex.get_level_values()
hmm = hmm.droplevel('Name', axis=1)

# Histogram of log returns (DOLLARBARS)
# Need masks because log returns are ridicolous when data switches from DAX to FTSE for example
mask1 = IndexData20Min_varying_DB.logreturn > -0.2
mask2 = IndexData20Min_varying_DB.logreturn < 0.2
plt.hist(IndexData20Min_varying_DB.logreturn, bins=np.arange(min(IndexData20Min_varying_DB[mask1].logreturn), max(IndexData20Min_varying_DB[mask2].logreturn), 0.001))

plt.close()

# Histogram of log returns (NON-DOLLARBARS)
histogramDat = IndexDat20Min
histogramDat = histogramDat[histogramDat.Volume > 0]
histogramDat = histogramDat.reset_index(drop=True)
DAXidx = histogramDat.Name == "DAX"
histogramDat['logreturn'] = np.log(histogramDat.Close/histogramDat.Close.shift(1)).fillna(0)
mask1 = histogramDat.logreturn > -0.2
mask2 = histogramDat.logreturn < 0.2

plt.hist(histogramDat[DAXidx].logreturn, bins=np.arange(min(histogramDat[mask1].logreturn), max(histogramDat[mask2].logreturn), 0.001))
plt.close()

# Looks like a more narrow normal distribution.. Hmm. No matter what LSTM processes information.
# So trying to feed information to the model in another way is interesting

x = torch.randn(2, 3, 4)
y = torch.softmax(x, dim=2)
y.sum(dim=1)


rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
what = output[:,-1, :]
what2 = output

hmm = dataframe.loc[0:train_to_idx,(slice(None), 'S&P')].droplevel('Name', axis=1)
hmm['return'] = (hmm.Close - hmm.Close.shift(-10)).fillna(0)
sum(hmm.loc[:,'return'])

# Lasse histogram log returns hong kong
HKidx = IndexDat.Name == "HK"
HongKong = IndexDat[HKidx]
VolumeMask = HongKong.Volume != 0
HongKong = HongKong[VolumeMask]
mask = HongKong.logreturn < 0.5
plt.hist(HongKong.logreturn, bins=np.arange(min(HongKong.logreturn), max(HongKong[mask].logreturn), 0.001))
plt.title("Sampled every 20 minutes (HK idx)", fontsize = 12)
plt.suptitle("Distribution of log returns", fontsize=16, fontweight='bold')
plt.close()

# Lasse histogram log returns hong kong dollarbars
HKidx = IndexData20Min_varying_DB.Name == "HK"
HongKong = IndexData20Min_varying_DB[HKidx]
VolumeMask = HongKong.Volume != 0
HongKong = HongKong[VolumeMask]
mask = HongKong.logreturn < 0.5
plt.hist(HongKong.logreturn, bins=np.arange(min(HongKong.logreturn), max(HongKong[mask].logreturn), 0.005))
plt.title("Sampled everytime x amount has changed hands (HK idx)", fontsize = 12)
plt.suptitle("Distribution of log returns", fontsize=16, fontweight='bold')
plt.close()



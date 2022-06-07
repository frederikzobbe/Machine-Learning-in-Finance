# 1. Reading in packages
import pandas as pd
import os as os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import train_test_split
import lightgbm as lgb

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0:len(features_used)]  # NUMBER OF FEATURES
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def LSTM_train(dataset, train_to_date, features_used, look_back=20, size_hidden=7, learning_rate=0.005, num_epochs=1000):
    # Load the dataset:
    dataset.reset_index(drop=True, inplace=True)
    dataframe_full = pd.DataFrame(dataset)
    dataframe = pd.DataFrame(dataframe_full[features_used])
    dataset_used = dataframe.values
    dataset_used = dataset_used.astype('float32')

    # Get train data
    train_to_idx = dataset.index[(dataset['Year'] == int(train_to_date[0:4])) &
                                 (dataset['Month'] == int(train_to_date[5:7])) &
                                 (dataset['Day'] == int(train_to_date[-2:]))][0]

    train, test = dataset_used[0:train_to_idx, :], dataset_used[train_to_idx:len(dataset_used),:]

    # Scale the dataset (*after splitting)
    scaler_out, scaler_feat = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))
    train_price_scaled = scaler_out.fit_transform(train[:, :1])
    train_feat_scaled = scaler_feat.fit_transform(train[:, 1:])
    test_price_scaled = scaler_out.transform(test[:, :1])
    test_feat_scaled = scaler_feat.transform(test[:, 1:])
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
            self.linear = nn.Linear(hidden_size, 1)

        def forward(self, x):
            x, _ = self.lstm(x)  # Run LSTM and store the hidden layer outputs
            x = x[:, -1, :]  # take the last hidden layer
            x = self.linear(x)  # a normal dense layer
            return x

    net = Net()

    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        prediction = net(trainX)
        loss = torch.sum((prediction.flatten() - trainY.flatten()) ** 2)
        progress_bar.set_description(f'Loss = {float(loss)}')
        loss.backward()
        opt.step()
        opt.zero_grad()

    # make predictions
    with torch.no_grad():
        trainPredict = net(trainX).numpy()
        testPredict = net(testX).numpy()

    # invert predictions
    trainPredict = scaler_out.inverse_transform(trainPredict)
    trainY_inv = scaler_out.inverse_transform([trainY.numpy()])
    testPredict = scaler_out.inverse_transform(testPredict)
    testY_inv = scaler_out.inverse_transform([testY.numpy()])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY_inv[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    return net

def LSTM_predict(dataset, model, features_used, predict_from_day, look_back=20):
    # Load the dataset:
    dataset.reset_index(drop=True, inplace=True)
    dataframe_full = pd.DataFrame(dataset)
    dataframe = pd.DataFrame(dataframe_full[features_used])
    dataset_used = dataframe.values
    dataset_used = dataset_used.astype('float32')

    # Get test data
    predict_from_idx = dataset.index[(dataset['Year'] == int(predict_from_day[0:4])) &
                                     (dataset['Month'] == int(predict_from_day[5:7])) &
                                     (dataset['Day'] == int(predict_from_day[-2:]))][0]

    train, test = dataset_used[0:predict_from_idx, :], dataset_used[predict_from_idx:len(dataset_used),:]

    # Scale the dataset (*after splitting)
    scaler_out, scaler_feat = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))
    train_price_scaled = scaler_out.fit_transform(train[:, :1])
    train_feat_scaled = scaler_feat.fit_transform(train[:, 1:])
    test_price_scaled = scaler_out.transform(test[:, :1])
    test_feat_scaled = scaler_feat.transform(test[:, 1:])
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

    # make predictions (recursively)
    # Features
    pred_feat = pd.DataFrame()
    for i in np.arange(len(features_used) - 1):
        # Create pandas dataframe of size (lookback+1)*length of train
        df = pd.DataFrame(columns=range(look_back + 1), index=range(len(train) - look_back))
        # Insert first column (our Y-vector)
        df.iloc[:, :1] = train[look_back:, i + 1].reshape(-1, 1)
        # Insert the previous look_back values (X-matrix)
        for j in np.arange(len(df)):
            df.iloc[j, 1:look_back + 1] = train[j:j + look_back, i + 1]

        X_train = df.iloc[:, 1:].astype(float)
        y_train = df.iloc[:, :1].astype(float)
        # Prepare data for model
        input_train, input_val, truth_train, truth_val = train_test_split(X_train, y_train, test_size=0.2,
                                                                          random_state=42)
        lgb_train = lgb.Dataset(input_train, truth_train)
        lgb_eval = lgb.Dataset(input_val, truth_val, reference=lgb_train)
        params = {'num_leaves': 100, 'learning_rate': 0.005, 'max_depth': 20, "verbosity": -1}

        # Train the model:
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=2000,
                        valid_sets=lgb_eval,
                        early_stopping_rounds=200,
                        verbose_eval=100)

        # Make predictions (on val):
        pred_gbm = gbm.predict(input_val, num_iteration=gbm.best_iteration)

        testScore = math.sqrt(mean_squared_error(truth_val, pred_gbm))
        print('Test Score (feature %.0f): %.2f RMSE' % (i + 1, testScore))

        # Make predictions (on future):
        pred_recursive_test = trainX[-1, :, i + 1].numpy().reshape(-1, 1)

        for k in np.arange(testX.shape[0]):
            input = pd.DataFrame(pred_recursive_test[-look_back:].reshape(1, -1))
            input.columns = pd.RangeIndex(start=1, stop=look_back + 1, step=1)
            tmp_out = gbm.predict(input, num_iteration=gbm.best_iteration).reshape(1, -1)
            pred_recursive_test = np.concatenate((pred_recursive_test, tmp_out), axis=0)

        pred_feat = pd.concat([pred_feat, pd.DataFrame(pred_recursive_test[look_back:])], axis=1)
        print('Feature %s is done' % (i + 1))

    # Prices
    with torch.no_grad():
        pred_recursive_test = trainX[-1, :, :].numpy()
        for i in np.arange(testX.shape[0]):
            pred_recursive_test = np.reshape(pred_recursive_test, (1, look_back + i, len(features_used)))
            input = pred_recursive_test[:, -look_back:, :]
            input_torch = torch.tensor(input, dtype=torch.float)
            next_price = net(input_torch).numpy().reshape(1, -1)
            next_feat = pred_feat.iloc[i, :].values.reshape(1, -1)
            tmp_out = np.concatenate((next_price, next_feat), axis=1)
            pred_recursive_test = np.concatenate(
                (pred_recursive_test.reshape(look_back + i, len(features_used)), tmp_out), axis=0)

    # invert predictions
    pred_recursive_test_inv = scaler_out.inverse_transform(pred_recursive_test[look_back - 1:, :1])

    return pred_recursive_test_inv

# Data must have columns 'Year', 'Month', 'Day' and the ones specified in features_used.
daxdatah = idxdatah[idxdatah['Name'] == 'DAX']

train_to_date = '2020-12-31'
features_used = ['Close', 'Volume', 'Hour', 'ROC-5', 'ROC-20',
       'EMA-10', 'EMA-200', 'Moterbike and car <3m', 'Car 3-6m', 'Total', '00 CPI Total', '01.1 Food']

dax_net = LSTM_train(dataset=daxdatah, train_to_date=train_to_date, features_used=features_used,
                     look_back=20, size_hidden=7, learning_rate=0.005, num_epochs=100)

pred_dax = LSTM_predict(dataset=daxdatah, model=dax_net, features_used=features_used, predict_from_day=train_to_date, look_back=20)

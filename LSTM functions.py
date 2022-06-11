# Navn: LSTM functions

################### DESCRIPTION #############################
# Programmet indeholder funktioner til at træne og predicte #
# fra en LSTM model                                         #
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

# Function: Create dataset for a pytorch
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0:len(features_used)]  # NUMBER OF FEATURES
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Function: Training the LSTM model (on minute data)
def LSTM_train(dataset, train_to_date, features_used, look_back=20, size_hidden=7, learning_rate=0.005, num_epochs=1000, pen_negativity_factor=1.5):
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
            x = x[:, -1, :]      # take the last hidden layer
            x = self.linear(x)   # a normal dense layer
            return x

    net = Net()

    #loss_fn = torch.sum(diff[diff >= 0] - 1.5*diff[diff <= 0])
    # old loss = torch.sum((prediction.flatten() - trainY.flatten() ** 2)

    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
    progress_bar = tqdm(range(num_epochs))
    for epoch in progress_bar:
        prediction = net(trainX)
        diff = (prediction.flatten() - trainY.flatten())
        loss = torch.sum(diff[diff >= 0] ** 2) + pen_negativity_factor*torch.sum(diff[diff < 0] ** 2)
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

# Function: Predicting from the LSTM model (on minute data)
def LSTM_predict(dataset, model, features_used, predict_from_day, look_back=20):
    
    # Sets a timer
    starttime = time.time()

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

    # make predictions (recursively) on features
    pred_feat = pd.DataFrame()
    for i in np.arange(len(features_used) - 1):
        # Create pandas dataframe of size (lookback+1)*length of train
        df = pd.DataFrame(columns=range(look_back + 1), index=range(len(train) - look_back))
        # Insert first column (our Y-vector)
        df.iloc[:, :1] = train[look_back:, i + 1].reshape(-1, 1)
        # Insert the previous look_back values (X-matrix)
        for j in np.arange(len(df)):
            df.iloc[j, 1:look_back + 1] = train[j:j + look_back, i + 1]

        # Prepare data for model
        X_train = df.iloc[:, 1:].astype(float)
        y_train = df.iloc[:, :1].astype(float)
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
                        verbose_eval=-1)

        # Make predictions (on future):
        pred_recursive_test = train[-look_back:,i+1].reshape(-1, 1)
        for k in np.arange(10*look_back):
            input = pd.DataFrame(pred_recursive_test[-look_back:].reshape(1, -1))
            input.columns = pd.RangeIndex(start=1, stop=look_back + 1, step=1)
            tmp_out = gbm.predict(input, num_iteration=gbm.best_iteration).reshape(1, -1)
            pred_recursive_test = np.concatenate((pred_recursive_test, tmp_out), axis=0)

        pred_feat = pd.concat([pred_feat, pd.DataFrame(pred_recursive_test[look_back:])], axis=1)
        print('Feature %s is done' % (i + 1))

    # Preddictions of prices
    with torch.no_grad():
        pred_recursive_test = train[-look_back:,:]
        for i in np.arange(10*look_back):
            pred_recursive_test = np.reshape(pred_recursive_test, (1, look_back + i, len(features_used)))
            input = pred_recursive_test[:, -look_back:, :]
            with torch.no_grad():
                input_torch = torch.tensor(input, dtype=torch.float)
                next_price = model(input_torch).numpy().reshape(1, -1)
            next_feat = pred_feat.iloc[i, :].values.reshape(1, -1)
            tmp_out = np.concatenate((next_price, next_feat), axis=1)
            pred_recursive_test = np.concatenate((pred_recursive_test.reshape(look_back + i, len(features_used)), tmp_out), axis=0)

    # invert predictions
    pred_recursive_test_inv = scaler_out.inverse_transform(pred_recursive_test[look_back - 1:, :1])
    print('Done predicting!')

    # Ends the timer
    endtime = time.time()
    dur = endtime - starttime
    print(' --- The function LSTM_predict took %s minutes to run ---' % (round(dur/60,2)) )

    return pred_recursive_test_inv

# Function: Predicting from the LSTM model (on minute data) with recalibrating 
def LSTM_predict_recal(dataset, model, features_used, predict_from_day, look_back=20):
    
    # Sets a timer
    starttime = time.time()
    
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

    #dataframe.iloc[0:predict_from_idx, 2].tail(30)
    #dataframe.iloc[predict_from_idx:,2].head(30)
    #pd.DataFrame(scaler_feat.inverse_transform(pred_feat)[:50])

    # Scale the dataset (*after splitting)
    scaler_out, scaler_feat = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))
    train_price_scaled = scaler_out.fit_transform(train[:, :1])
    train_feat_scaled = scaler_feat.fit_transform(train[:, 1:])
    test_price_scaled = scaler_out.transform(test[:, :1])
    test_feat_scaled = scaler_feat.transform(test[:, 1:])
    train = np.column_stack((train_price_scaled, train_feat_scaled))
    test = np.column_stack((test_price_scaled, test_feat_scaled))

    # make predictions (recursively) on features
    pred_feat = pd.DataFrame()
    for i in np.arange(len(features_used) - 1):
        # Create pandas dataframe of size (lookback+1)*length of train
        df = pd.DataFrame(columns=range(look_back + 1), index=range(len(train) - look_back))
        # Insert first column (our Y-vector)
        df.iloc[:, :1] = train[look_back:, i + 1].reshape(-1, 1)
        # Insert the previous look_back values (X-matrix)
        for j in np.arange(len(df)):
            df.iloc[j, 1:look_back + 1] = train[j:j + look_back, i + 1]

        # Prepare data for model
        X_train = df.iloc[:, 1:].astype(float)
        y_train = df.iloc[:, :1].astype(float)
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
                        verbose_eval=-1)

        # Make predictions (on future):
        pred_recursive_test = train[-look_back:,i+1].reshape(-1, 1)
        
        dates = dataset['CET'][predict_from_idx:]    
        dates = np.unique([str(date)[:10] for date in dates])

        today_idx = predict_from_idx
        for d in np.arange(len(dates)-1):
            nextday = dates[d+1]
            nextday_idx = dataset.index[(dataset['Year'] == int(nextday[0:4])) &
                                     (dataset['Month'] == int(nextday[5:7])) &
                                     (dataset['Day'] == int(nextday[-2:]))][0]
            
            houridxs = nextday_idx - today_idx

            pred_shift_test = scaler_feat.transform(dataset_used[today_idx-look_back:today_idx, 1:])[:,i].reshape(-1,1)

            for k in np.arange(houridxs):
                input = pd.DataFrame(pred_shift_test[-look_back:].reshape(1, -1))
                input.columns = pd.RangeIndex(start=1, stop=look_back + 1, step=1)
                tmp_out = gbm.predict(input, num_iteration=gbm.best_iteration).reshape(1, -1)
                pred_shift_test = np.concatenate((pred_shift_test, tmp_out), axis=0)

                #len(pred_shift_test)
            
            pred_recursive_test = np.concatenate((pred_recursive_test, pred_shift_test[look_back:]), axis=0)
            today_idx = nextday_idx

        pred_feat = pd.concat([pred_feat, pd.DataFrame(pred_recursive_test[look_back:])], axis=1)
        print('Feature %s is done' % (i + 1))

    # Predictions of prices
    dates = dataset['CET'][predict_from_idx:]    
    dates = np.unique([str(date)[:10] for date in dates])

    today_idx = predict_from_idx

    pred_recursive_test = train[-look_back:,:]
    for d in np.arange(len(dates)-1):
        nextday = dates[d+1]
        nextday_idx = dataset.index[(dataset['Year'] == int(nextday[0:4])) &
                                     (dataset['Month'] == int(nextday[5:7])) &
                                     (dataset['Day'] == int(nextday[-2:]))][0]
            
        houridxs = nextday_idx - today_idx

        pred_shift_feat = scaler_feat.transform(dataset_used[today_idx-look_back:today_idx, 1:])
        pred_shift_out = scaler_out.transform(dataset_used[today_idx-look_back:today_idx, :1])
        pred_shift_test = np.concatenate((pred_shift_out, pred_shift_feat), axis=1)

        for i in np.arange(houridxs):
            pred_shift_test = np.reshape(pred_shift_test, (1, look_back + i, len(features_used)))
            input = pred_shift_test[:, -look_back:, :]
            with torch.no_grad():
                input_torch = torch.tensor(input, dtype=torch.float)
                next_price = model(input_torch).numpy().reshape(1, -1)
            next_feat = pred_feat.iloc[today_idx-predict_from_idx+i, :].values.reshape(1, -1)
            tmp_out = np.concatenate((next_price, next_feat), axis=1)
            pred_shift_test = np.concatenate((pred_shift_test.reshape(look_back + i, len(features_used)), tmp_out), axis=0)

        pred_recursive_test = np.concatenate((pred_recursive_test, pred_shift_test[look_back:]), axis=0)
        today_idx = nextday_idx

    # invert predictions
    pred_recursive_test_inv = scaler_out.inverse_transform(pred_recursive_test[look_back - 1:, :1])
    print('Done predicting!')

    # Ends the timer
    endtime = time.time()
    dur = endtime - starttime
    print(' --- The function LSTM_predict_recal took %s minutes to run ---' % (round(dur/60,2)) )

    return pred_recursive_test_inv

# Function: Plotting predictions against true data
def LSTM_plot(dataset, predictions, predict_from_day, features_used, look_back, size_hidden, num_epochs, pen_negativity_factor, save, name, view = 20):
    # Load the dataset:
    dataset.reset_index(drop=True, inplace=True)
    dataframe_full = pd.DataFrame(dataset)
    dataframe = pd.DataFrame(dataframe_full[features_used])
    dataset_used = dataframe.values
    dataset_used = dataset_used.astype('float32')

    splitpoint = dataset.index[(dataset['Year'] == int(predict_from_day[0:4])) &
                                (dataset['Month'] == int(predict_from_day[5:7])) &
                                (dataset['Day'] == int(predict_from_day[-2:]))][0]

    len(predictions)
    testPredictPlot = np.empty_like(dataset_used[:, :1])
    testPredictPlot[:, :] = np.nan
    testPredictPlot[splitpoint-1:splitpoint+len(predictions)-1, :] = predictions
    #splitdate = dataframe_full[dataframe_full.index == splitpoint]['CET'].dt.date.item().strftime('%d %b %Y')

    xrangemin, xrangemax = splitpoint-view, splitpoint+2*view
    dates_between = dataset['CET'][xrangemin:xrangemax]
    dates_between = np.unique([str(date)[:10] for date in dates_between])

    #date_index = pd.date_range(start=x_start_date, end=x_end_date, freq="D")
    #date_index = [str(date)[:10] for date in date_index]

    x_ticks = np.linspace(start=xrangemin, stop=xrangemax, num=len(dates_between))

    # Short period (After traindata ends)
    dpi = 500 if save == 'yes' else 130
    fig, ax = plt.subplots(figsize=(8, 5), dpi = dpi)
    plt.plot(dataset_used[:, :1], label='Observations', color = 'b')
    plt.plot(testPredictPlot, label='Predict: Test', color = 'r')
    ax.legend(loc='upper left', frameon=False)
    plt.title(str(dataset['Name'][0]) + ' ' + str(dataset['Type'][0]) + ' predictions \n ' + 'Lookback: ' + str(look_back) + ', Hidden states: '+ str(size_hidden) + ', Epochs: ' + str(num_epochs) + ', Penalty: ' + str(pen_negativity_factor) )
    #fig.suptitle('This sentence is\nbeing split\ninto three lines')
    plt.axvline(x = splitpoint-1, color = 'r', linestyle = '-')
    #plt.text(splitpoint+200,8000, str(splitdate),rotation=0)
    plt.xlim([xrangemin, xrangemax])
    plt.xticks(x_ticks, dates_between, rotation=30)
    plt.ylim([min(dataset_used[xrangemin:xrangemax,0])*0.95, max(dataset_used[xrangemin:xrangemax,0])*1.05])

    plt.savefig(str(name) + '.png') if (save == 'yes') else plt.show()

# Function: Evaluating predictions from true data
def pred_eval(dataset, predictions, predict_from_day, days):
    dataset.reset_index(drop=True, inplace=True)
    splitpoint = dataset.index[(dataset['Year'] == int(predict_from_day[0:4])) &
                                (dataset['Month'] == int(predict_from_day[5:7])) &
                                (dataset['Day'] == int(predict_from_day[-2:]))][0]
    
    dates = dataset['CET'][splitpoint:]
    datesunique = np.unique([str(date)[:10] for date in dates])
    nextday = datesunique[days]

    eval_idx = dataset.index[(dataset['Year'] == int(nextday[0:4])) &
                                     (dataset['Month'] == int(nextday[5:7])) &
                                     (dataset['Day'] == int(nextday[-2:]))][0]

    print(' --- Evaluating ' + str(dataset['Name'][0]) + ' ' + str(dataset['Type'][0]) + ' from ' + 
            str(predict_from_day) + ' to ' + str(dataset['CET'][eval_idx])[:10] )
    
    pred = predictions[eval_idx - splitpoint]
    obs  = dataset['Close'][eval_idx]

    columns = ['Price: ' + str(predict_from_day), 'Exp. price: ' + str(dataset['CET'][eval_idx])[:10], 
                'Act. price: ' + str(dataset['CET'][eval_idx])[:10], 'Exp. rate of return', 'Act. rate of return', 'Difference in return', 'Pct. difference in price']
    len(columns)
    result = pd.DataFrame(columns=columns)

    result[columns[0]]  = predictions[0]
    result[columns[1]]  = pred
    result[columns[2]]  = obs
    result[columns[3]]  = ((pred-predictions[0])/predictions[0]*100).item()
    result[columns[4]]  = ((obs-predictions[0])/predictions[0]*100).item()
    result[columns[5]]  = result.iloc[0,3] - result.iloc[0,4]
    result[columns[6]]  = (pred-obs)/obs*100    

    return result

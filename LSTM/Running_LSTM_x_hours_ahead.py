
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

# Function: Create dataset for a pytorch
def create_dataset(dataset, look_back=1, hours_ahead=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-hours_ahead):
        a = dataset[i:(i+look_back), 0:len(features_used)]  # NUMBER OF FEATURES
        dataX.append(a)
        dataY.append(dataset[i + look_back + hours_ahead - 1, 0])
    return np.array(dataX), np.array(dataY)

valutadatahour['Name'].value_counts()
dataset      = valutadatahour[valutadatahour['Name'] == 'ETHER']
dataset.columns.sort_values()

df = dataset.copy()
df.index = df["CET"]
df.sort_index(inplace=True)
df['Close'].plot()
plt.axvline(x='2020-02-01')
plt.axvline(x='2020-06-01')
plt.axvline(x='2021-04-01')
plt.show()

train_end_date  = '2020-01-31'
val_start_date  = '2020-06-01'
test_start_date = '2021-04-01'
test_end_date   = '2022-04-29'

features_all = list(['Close']) + list(set(list(dataset.columns)) - set(list(['Open', 'High', 'Low', 'Name', 'Type', 'CET', 'Minute', 'Close'])))
features_tmp =['Close', 'Volume', 'Hour', 'ROC-5', 'ROC-20', 'EMA-5', 'EMA-50', 'EMA-200']

look_back                  = 40
size_hidden                = 100
learning_rate              = 0.005
num_epochs                 = 100
pen_negativity_factor      = 1
hours_ahead                = 8
features_used              = features_tmp

# Load the dataset:
dataset.reset_index(drop=True, inplace=True)
dataframe_full = pd.DataFrame(dataset)
dataframe = pd.DataFrame(dataframe_full[features_used])
dataset_used = dataframe.values
dataset_used = dataset_used.astype('float32')

# Get train data
train_end_idx = dataset.index[(dataset['Year'] == int(train_end_date[0:4])) &
                             (dataset['Month'] == int(train_end_date[5:7])) &
                             (dataset['Day'] == int(train_end_date[-2:]))][0]

val_start_idx = dataset.index[(dataset['Year'] == int(val_start_date[0:4])) &
                             (dataset['Month'] == int(val_start_date[5:7])) &
                             (dataset['Day'] == int(val_start_date[-2:]))][0]

test_start_idx = dataset.index[(dataset['Year'] == int(test_start_date[0:4])) &
                             (dataset['Month'] == int(test_start_date[5:7])) &
                             (dataset['Day'] == int(test_start_date[-2:]))][0]

train, val, test = dataset_used[0:train_end_idx, :], dataset_used[val_start_idx:test_start_idx,:], dataset_used[test_start_idx:len(dataset_used),:]

# Scale the dataset (*after splitting)
scaler_out, scaler_feat = MinMaxScaler(feature_range=(0, 1)), MinMaxScaler(feature_range=(0, 1))
train_price_scaled = scaler_out.fit_transform(train[:, :1])
train_feat_scaled = scaler_feat.fit_transform(train[:, 1:])
val_price_scaled = scaler_out.transform(val[:, :1])
val_feat_scaled = scaler_feat.transform(val[:, 1:])
test_price_scaled = scaler_out.transform(test[:, :1])
test_feat_scaled = scaler_feat.transform(test[:, 1:])
train = np.column_stack((train_price_scaled, train_feat_scaled))
val   = np.column_stack((val_price_scaled, val_feat_scaled))
test  = np.column_stack((test_price_scaled, test_feat_scaled))

# Reshape into X = t and Y = t + 1
trainX, trainY  = create_dataset(train, look_back, hours_ahead=hours_ahead)
valX, valY      = create_dataset(val, look_back, hours_ahead=hours_ahead)
testX, testY    = create_dataset(test, look_back, hours_ahead=hours_ahead)

# Reshape input to be [samples, time steps, features]
trainX  = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], len(features_used)))
valX    = np.reshape(valX, (valX.shape[0], valX.shape[1], len(features_used)))
testX   = np.reshape(testX, (testX.shape[0], testX.shape[1], len(features_used)))

trainX = torch.tensor(trainX, dtype=torch.float)
trainY = torch.tensor(trainY, dtype=torch.float)
valX = torch.tensor(valX, dtype=torch.float)
valY = torch.tensor(valY, dtype=torch.float)
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

# loss_fn = torch.sum(diff[diff >= 0] - 1.5*diff[diff <= 0])
# old loss = torch.sum((prediction.flatten() - trainY.flatten() ** 2)

val_scores = []
train_scores = []
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
    
    with torch.no_grad():
        valPredict = net(valX).numpy()
        trainPredict = net(trainX).numpy()
    valPredict = scaler_out.inverse_transform(valPredict)
    valY_inv = scaler_out.inverse_transform([valY.numpy()]) # Change to val dataset later on
    valScore = math.sqrt(mean_squared_error(valY_inv[0], valPredict[:, 0]))
    val_scores.append(valScore)

    trainPredict = scaler_out.inverse_transform(trainPredict)
    trainY_inv = scaler_out.inverse_transform([trainY.numpy()]) # Change to val dataset later on
    trainScore = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:, 0]))
    train_scores.append(trainScore)
    #print('Test Score: %.2f RMSE' % (testScore))

# make predictions
with torch.no_grad():
    trainPredict    = net(trainX).numpy()
    valPredict      = net(valX).numpy()
    testPredict     = net(testX).numpy()

# invert predictions
trainPredict    = scaler_out.inverse_transform(trainPredict)
trainY_inv      = scaler_out.inverse_transform([trainY.numpy()])
valPredict      = scaler_out.inverse_transform(valPredict)
valY_inv        = scaler_out.inverse_transform([valY.numpy()])
testPredict     = scaler_out.inverse_transform(testPredict)
testY_inv       = scaler_out.inverse_transform([testY.numpy()])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY_inv[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
valScore = math.sqrt(mean_squared_error(valY_inv[0], valPredict[:, 0]))
print('Val Score: %.2f RMSE' % (valScore))
testScore = math.sqrt(mean_squared_error(testY_inv[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

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
plt.title("Train and validation loss in LSTM NN \n " + 'Asset: ' + str(dataset['Name'][0]))
plt.legend()
plt.show()
#plt.savefig('TrainlossDAX.png')

## -------------------------------  PREDICT -----------------------------------
predict_from_idx = test_start_idx
model=net

starttime = time.time()

pred_recursive_test         = val[-(look_back+hours_ahead-1):,:]
pred_recursive_test_price   = pred_recursive_test[:,0]

for k in np.arange(len(test)):
    pred_shift_test_feat = scaler_feat.transform(dataset_used[predict_from_idx-look_back-hours_ahead+1+k:predict_from_idx-hours_ahead+1+k,1:])
    pred_shift_test_out = scaler_out.transform(dataset_used[predict_from_idx-look_back-hours_ahead+1+k:predict_from_idx-hours_ahead+1+k,:1])
    pred_shift_test = np.column_stack((pred_shift_test_out, pred_shift_test_feat))
    
    pred_shift_test = np.reshape(pred_shift_test, (1, look_back, len(features_used)))
    input = pred_shift_test[:, -look_back:, :]
    with torch.no_grad():
        input_torch = torch.tensor(input, dtype=torch.float)
        next_price = model(input_torch).numpy()
    pred_recursive_test_price = np.concatenate((pred_recursive_test_price, next_price[0]), axis=0)

# invert predictions
pred_recursive_test_inv = scaler_out.inverse_transform(pred_recursive_test_price[look_back+hours_ahead-2:].reshape(-1, 1))
print('Done predicting!')

# Ends the timer
endtime = time.time()
dur = endtime - starttime
print(' --- The function LSTM_predict_recal took %s minutes to run ---' % (round(dur/60,2)) )

# -----------------------------  PLOT  --------------------------------------

predictions=pred_recursive_test_inv
save='no'
name = 'plot'
view = 2000
dates_between_ticks = 100

splitpoint = test_start_idx

with torch.no_grad():
    val_predictions   = model(valX).numpy()
    train_predictions = model(trainX).numpy()

#trainX[0,:,1:3]
#scaler_feat.transform(dataset_used[:look_back,1:])[:,:2]
#dataset['CET'][:look_back]

train_predictions_inv = scaler_out.inverse_transform(train_predictions)
val_predictions_inv   = scaler_out.inverse_transform(val_predictions)

trainPredictPlot = np.empty_like(dataset_used[:, :1])
trainPredictPlot[:, :] = np.nan
valPredictPlot = np.empty_like(dataset_used[:, :1])
valPredictPlot[:, :] = np.nan
testPredictPlot = np.empty_like(dataset_used[:, :1])
testPredictPlot[:, :] = np.nan

for i in np.arange(len(train_predictions_inv[::hours_ahead])-1):
    trainPredictPlot[-1+look_back+(i+1)*hours_ahead:-1+look_back+(i+2)*hours_ahead] = train_predictions_inv[i*hours_ahead]
for i in np.arange(len(val_predictions_inv[::hours_ahead])-1):
    valPredictPlot[val_start_idx-1+look_back+(i+1)*hours_ahead:val_start_idx-1+look_back+(i+2)*hours_ahead] = val_predictions_inv[i*hours_ahead]
for i in np.arange(len(predictions[::hours_ahead])):
    testPredictPlot[splitpoint+i*hours_ahead:splitpoint+(i+1)*hours_ahead,:] = predictions[1+i*hours_ahead]

xrangemin, xrangemax = 0, len(testPredictPlot)
dates_between = dataset['CET'][xrangemin:xrangemax]
dates_between = np.unique([str(date)[:10] for date in dates_between])

x_ticks = np.linspace(start=xrangemin, stop=xrangemax, num=len(dates_between))

# Short period (After traindata ends)
dpi = 500 if save == 'yes' else 130
fig, ax = plt.subplots(figsize=(8, 5), dpi = dpi)
plt.plot(dataset_used[:, :1], label='Observations', color = 'blue')
plt.plot(trainPredictPlot, label='Predict: Train', color = 'orange')
plt.plot(valPredictPlot, label='Predict: Val', color = 'red')
plt.plot(testPredictPlot, label='Predict: Test', color = 'green')
ax.legend(loc='upper left', frameon=False)
plt.title(str(dataset['Name'][0]) + ' ' + str(dataset['Type'][0]) + ' predictions \n ' + 'Lookback: ' + str(look_back) + ', Hidden states: '+ str(size_hidden) + ', Epochs: ' + str(num_epochs) + ', Penalty: ' + str(pen_negativity_factor) )
#fig.suptitle('This sentence is\nbeing split\ninto three lines')
plt.axvline(x = train_end_idx-1, color = 'black', linestyle = '-')
plt.axvline(x = val_start_idx-1, color = 'black', linestyle = '-')
plt.axvline(x = splitpoint-1, color = 'black', linestyle = '-')
#plt.text(splitpoint+200,8000, str(splitdate),rotation=0)
plt.xlim([xrangemin, xrangemax])
plt.xticks(x_ticks[::dates_between_ticks], dates_between[::dates_between_ticks], rotation=30)
plt.ylim([min(dataset_used[xrangemin:xrangemax,0])*0.95, max(dataset_used[xrangemin:xrangemax,0])*1.05])

plt.savefig(str(name) + '.png') if (save == 'yes') else plt.show()


# ------------------------------- SAVE DATA ----------------------------------

print(' --- Evaluating ' + str(dataset['Name'][0]) + ' ' + str(dataset['Type'][0]) + ' with ' + 
         str(hours_ahead) + ' hours ahead predictions and look_back = ' + str(look_back))

times = dataset['CET'][splitpoint::hours_ahead]
pred =  testPredictPlot[splitpoint::hours_ahead]
obs =   dataset['Close'][splitpoint::hours_ahead]

columns = ['CET', 'Name', 'Close', 'Exp. price ' + str(hours_ahead) + ' hours ago', 'Diff']

result = pd.DataFrame(columns=columns)

result[columns[0]]  = times
result[columns[1]]  = dataset['Name'][0]
result[columns[2]]  = obs
result[columns[3]]  = pred
result[columns[4]]  = result[columns[3]] - result[columns[2]]

mean_pct_afv = np.round(np.mean(result['Diff'])/np.mean(result['Close'])*100,2)
print(' --- On average our predictions are ' + str(mean_pct_afv) + ' %' + ' away from the real prices ')
#--- On average our predictions are -0.27 % away from the real prices

np.mean(np.abs(result['Diff']))  # 110.18268819192836

result.to_csv('ETHER_8ha_pred_val_v1.txt', index = False, header = True)

os.chdir("/Users/mathiasfrederiksen/Desktop/Forsikringsmatematik/5. år/Applied Machine Learning/Data/SwissData/Predictions")












# CHANGES TO FIT LASSES DATAKRAV
os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData/Predictions")
daxdatah    = idxdatah[idxdatah['Name'] == 'DAX']
spdatah     = idxdatah[idxdatah['Name'] == 'S&P']
nasdaqdatah = idxdatah[idxdatah['Name'] == 'NASDAQ']
hkdatah     = idxdatah[idxdatah['Name'] == 'HK']
ftsedatah   = idxdatah[idxdatah['Name'] == 'FTSE']
eurusddatah   = valutadatahour[valutadatahour['Name'] == 'EUROUSD']
bitcoindatah  = valutadatahour[valutadatahour['Name'] == 'BITCOIN']
etherdatah    = valutadatahour[valutadatahour['Name'] == 'ETHER']
coffeedatah   = commodatahour[commodatahour['Name'] == 'COFFEE']
oildatah      = commodatahour[commodatahour['Name'] == 'OIL']
gasdatah      = commodatahour[commodatahour['Name'] == 'GAS']

tmp = pd.read_csv('GAS_8ha_pred_val_v1.txt', index_col=None, parse_dates=['CET'], engine='python')

dataset = gasdatah
dataset.reset_index(drop=True, inplace=True)
splitdate = test_start_date
splitpoint = dataset.index[(dataset['Year'] == int(splitdate[0:4])) &
                            (dataset['Month'] == int(splitdate[5:7])) &
                            (dataset['Day'] == int(splitdate[-2:]))][0]
dataset['CET'][splitpoint-1:splitpoint+1]

dates_between = dataset['CET'][:splitpoint]
dates_between = np.unique([str(date)[:10] for date in dates_between])
len(dates_between)
times, close = [], []

for d in dates_between:
    idx = dataset.index[(dataset['Year'] == int(d[0:4])) &
                                (dataset['Month'] == int(d[5:7])) &
                                (dataset['Day'] == int(d[-2:]))][0]
    times.append(dataset['CET'][idx])
    close.append(dataset['Close'][idx])

columns = ['CET', 'Name', 'Close', 'Exp. price ' + str(hours_ahead) + ' hours ago', 'Diff']

result = pd.DataFrame(columns=columns)

result[columns[0]]  = times
result[columns[1]]  = dataset['Name'][0]
result[columns[2]]  = close
result[columns[3]]  = np.nan
result[columns[4]]  = np.nan

result = pd.concat([result,tmp])

result[len(dates_between)-2:len(dates_between)+2]
result.reset_index(inplace=True, drop=True)

result.to_csv('GAS_8ha_pred_val_v2.txt', index = False, header = True)



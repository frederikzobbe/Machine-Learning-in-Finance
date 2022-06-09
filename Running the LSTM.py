# Navn: Running the LSTM

################### DESCRIPTION #############################
# Programmet skal være workspace for at køre LSTM modellen  #
#############################################################

# 1. Reading in packages
import pandas as pd
import numpy as np
import os as os

# Data must have columns 'Year', 'Month', 'Day' and the ones specified in features_used.
daxdatah = idxdatah[idxdatah['Name'] == 'DAX']
spdatah = idxdatah[idxdatah['Name'] == 'S&P']
hkh[['CET', 'Volume']].head(20)

train_to_date = '2021-01-28'
features_tmp = ['Close', 'Volume']
features_used = ['Close', 'Volume', 'Hour', 'ROC-5', 'ROC-20', 'EMA-10', 'EMA-200', 'Moterbike and car <3m', 
                     'Car 3-6m', 'Total', '00 CPI Total', '01.1 Food']

look_back = 50
dataset= spdatah

sp_net  = LSTM_train(dataset=dataset, train_to_date=train_to_date, features_used=features_used,look_back=look_back, size_hidden=8, learning_rate=0.005, num_epochs=1000)
pred_sp = LSTM_predict(dataset=dataset, model=sp_net, features_used=features_used, predict_from_day=train_to_date, look_back=look_back)
eval    = pred_eval(dataset=dataset, predictions = pred_sp, predict_from_day=train_to_date, days=1)

LSTM_plot(daxdatah, pred_dax, predict_from_day=train_to_date, features_used=features_used, look_back=look_back)

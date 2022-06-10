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
#spdatah = idxdatah[idxdatah['Name'] == 'S&P']
#hkh[['CET', 'Volume']].head(20)

daxdatad

train_to_date = '2021-06-30'
features_tmp = ['Close', 'Volume']
features_tmp = ['Close', 'Volume', 'ROC-5', 'ROC-20', 'EMA-10', 'EMA-200', '00 CPI Total']
features_used = ['Close', 'Volume', 'Hour', 'ROC-5', 'ROC-20', 'EMA-10', 'EMA-200', 'Moterbike and car <3m', 
                     'Car 3-6m', 'Total', '00 CPI Total', '01.1 Food']

look_back                   = 100
dataset                     = daxdatad
features_used               = features_tmp
size_hidden                 = 100
learning_rate               = 0.005
num_epochs                  = 200
pen_negativity_factor       = 1.5

dax_net  =        LSTM_train(dataset=dataset, train_to_date=train_to_date, features_used=features_used,look_back=look_back, size_hidden=size_hidden, learning_rate=learning_rate, num_epochs=num_epochs, pen_negativity_factor=pen_negativity_factor)
pred_dax =        LSTM_predict(dataset=dataset, model=dax_net, features_used=features_used, predict_from_day=train_to_date, look_back=look_back)
pred_dax_update = LSTM_predict_recal(dataset=dataset, model=dax_net, features_used=features_used, predict_from_day=train_to_date, look_back=look_back)
eval    =         pred_eval(dataset=dataset, predictions = pred_dax, predict_from_day=train_to_date, days=1)

LSTM_plot(save = 'no', name = 'plot', dataset = dataset, predictions=pred_dax, predict_from_day=train_to_date, features_used=features_used, look_back = look_back, size_hidden=size_hidden, num_epochs=num_epochs, pen_negativity_factor=pen_negativity_factor, view=200)
#os.getcwd() # Place for saving figures
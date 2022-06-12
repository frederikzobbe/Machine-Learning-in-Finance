# Navn: Modifying the data
# Oprettet: 25-05-2022
# Senest ændret: 30-05-2022

################### CHANGELOG ###########################
# FZC: Created the document                             #
################### DESCRIPTION #########################
# This program is modifying the initial data from the   #
# swiss bank. Primarily the data's timestamp is changed #
#########################################################

# 1. Reading in packages
from operator import index
import pandas as pd
import numpy as np
import os as os
import time as time
import datetime as dt
from dateutil import parser
import pytz
from pytz import all_timezones
from datetime import datetime, timedelta
import operator
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.mode.chained_assignment = None  # default='warn'

# 2. Creating functions

# FUNCTION: Adds time and date columns to data
def timefunc(x: pd.DataFrame, colint: int):
    
    # Sets a timer
    starttime = time.time()

    # tmp column
    x['tmptime'] = x['Local time'].str[:-13]
    
    # Converts column to datetime
    tmp  = pd.to_datetime(x['tmptime'], format= "%d.%m.%Y %H:%M:%S")

    # Constructs new columns
    x['CET'] = tmp
    x['Year']   = tmp.dt.year
    x['Month']  = tmp.dt.month
    x['Day']    = tmp.dt.day
    x['Hour']   = tmp.dt.hour
    x['Minute'] = tmp.dt.minute

    # Delete initial column
    x.drop(x.columns[colint], axis=1, inplace = True)
    x.drop("tmptime", axis=1, inplace=True)

    # Ends the timer
    endtime = time.time()
    dur = endtime - starttime
    print(' --- The function TIMEFUNC took %s seconds to run ---' %round(dur,2))
    return x

# 3. Reading in data
os.chdir("/Users/frederikzobbe/Desktop/Data")
#os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")

#os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")
df1 = pd.read_csv('BTCUSD_Candlestick_1_Hour_BID_08.05.2017-30.04.2022.csv')
BIT = df1
BIT['Name'] = "BITCOIN"
BIT['Type'] = "Valuta"
df2 = pd.read_csv('ETHUSD_Candlestick_1_Hour_BID_12.12.2017-30.04.2022.csv')
ETH = df2
ETH['Name'] = "ETHER"
ETH['Type'] = "Valuta"
df3 = pd.read_csv('EURUSD_Candlestick_1_Hour_BID_01.01.2013-30.04.2022.csv')
EUR = df3
EUR['Name'] = "EUROUSD"
EUR['Type'] = "Valuta"
data = pd.concat([BIT, ETH, EUR])

data = timefunc(data, 0)
data.to_csv('ValutaDataHour.txt', index = False, header=True)


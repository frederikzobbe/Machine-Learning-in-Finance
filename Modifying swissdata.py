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
df1 = pd.read_csv('DEU.IDXEUR_Candlestick_1_Hour_BID_01.01.2018-30.04.2022.csv')
DAX = df1
DAX['Name'] = "DAX"
DAX['Type'] = "Index"
df2 = pd.read_csv('USA500.IDXUSD_Candlestick_1_Hour_BID_01.01.2018-30.04.2022.csv')
SP = df2
SP['Name'] = "S&P"
SP['Type'] = "Index"
df3 = pd.read_csv('USATECH.IDXUSD_Candlestick_1_Hour_BID_01.01.2018-30.04.2022.csv')
NAS = df3
NAS['Name'] = "NASDAQ"
NAS['Type'] = "Index"
df4 = pd.read_csv('HKG.IDXHKD_Candlestick_1_Hour_BID_01.01.2018-30.04.2022.csv')
HK = df4
HK['Name'] = "HK"
HK['Type'] = "Index"
df5 = pd.read_csv('GBR.IDXGBP_Candlestick_1_Hour_BID_01.01.2018-30.04.2022.csv')
FTSE = df5
FTSE['Name'] = "FTSE"
FTSE['Type'] = "Index"

#os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")
#date_cols_tmp = ['Local time']
#df1 = pd.read_csv('SwissData.txt', index_col=None, parse_dates=date_cols_tmp)

# 4. Applying functions to data
df = pd.concat([DAX, HK, SP, NAS, FTSE], ignore_index=True)
data = timefunc(df, 0)
data['Name'].value_counts()
data.to_csv('IndexDataHour.txt', index = False, header=True)

# 4.5
os.chdir("/Users/frederikzobbe/Desktop/Data")
data = timefunc(df, 0)

# 5. Test of results
ticker = 'DAX'
df_sub = swissdata[(swissdata['Name'] == ticker) & (swissdata['Year'] < 2019) & (swissdata['Month'] < 3)]
df_sub.head(5)
len(df_sub)

x = df_sub['Local time']
y = df_sub['Close']
z = df_sub['CET']

plt.subplot(1,2,1)
plt.plot(x,y,'r')
plt.subplot(1,2,2)
plt.plot(z,y,'b')
plt.show()

import plotly.express as px
fig = px.line(df_sub, x="CET", y="Close", color='Name')
fig.show()

## ---------------------------- Michael Modifying 20 min index data ----------------------------
os.chdir("/Users/mikki/Desktop/AppML/New env Personal projects/Final project/20minIndex")

df1 = pd.read_csv('DEU.IDXEUR_Candlestick_20_M_BID_01.01.2018-31.12.2019.csv')
df2 = pd.read_csv('DEU.IDXEUR_Candlestick_20_M_BID_01.01.2020-30.04.2022.csv')

DAX = pd.concat([df1, df2]).reset_index(drop = True)
DAX['Name'] = "DAX"
DAX['Type'] = "Index"

df1 = pd.read_csv('HKG.IDXHKD_Candlestick_20_M_BID_01.01.2018-31.12.2019.csv')
df2 = pd.read_csv('HKG.IDXHKD_Candlestick_20_M_BID_01.01.2020-30.04.2022.csv')
HK = pd.concat([df1, df2]).reset_index(drop = True)
HK['Name'] = "HK"
HK['Type'] = "Index"

df1 = pd.read_csv('USA500.IDXUSD_Candlestick_20_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('USA500.IDXUSD_Candlestick_20_M_BID_01.01.2021-30.04.2022.csv')
SP = pd.concat([df1, df2]).reset_index(drop = True)
SP['Name'] = "S&P"
SP['Type'] = "Index"

df1 = pd.read_csv('USATECH.IDXUSD_Candlestick_20_M_BID_01.01.2018-31.12.2019.csv')
df2 = pd.read_csv('USATECH.IDXUSD_Candlestick_20_M_BID_01.01.2020-30.04.2022.csv')
NAS = pd.concat([df1, df2]).reset_index(drop = True)
NAS['Name'] = "NASDAQ"
NAS['Type'] = "Index"

df1 = pd.read_csv('GBR.IDXGBP_Candlestick_20_M_BID_01.01.2018-31.12.2019.csv')
df2 = pd.read_csv('GBR.IDXGBP_Candlestick_20_M_BID_01.01.2020-30.04.2022.csv')
FTSE = pd.concat([df1, df2]).reset_index(drop = True)
FTSE['Name'] = "FTSE"
FTSE['Type'] = "Index"


# I Need to add 1 hour to my data in order to ensure time consistency with our other datasets.
FTSE.dtypes
IndexdataHour_DB.dtypes

IndexDat20Min = pd.concat([DAX, FTSE, SP, NAS, HK]).reset_index(drop = True)
IndexDat20Min.dtypes
IndexDat20Min['Gmt time'] = pd.to_datetime(IndexDat20Min['Gmt time'].str[:-4], format= "%d.%m.%Y %H:%M:%S")
IndexDat20Min['CET'] = IndexDat20Min['Gmt time'] + pd.DateOffset(hours = 1)
IndexDat20Min["Year"] = IndexDat20Min.CET.dt.year
IndexDat20Min['Month'] = IndexDat20Min.CET.dt.month
IndexDat20Min['Day'] = IndexDat20Min.CET.dt.day

os.chdir("/Users/mikki/Desktop/AppML/New env Personal projects/Final project/data/SwissData")
IndexDat20Min.to_csv("20MinIndexData", index = None, index_label=None)

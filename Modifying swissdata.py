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

# 2. Reading in data
os.chdir("/Users/frederikzobbe/Desktop/Data")
#os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")

#df = pd.read_csv('SwissData.txt', index_col=None, header = None)

df1 = pd.read_csv('DEU.IDXEUR_Candlestick_1_M_BID_01.01.2018-31.12.2018.csv')
df2 = pd.read_csv('DEU.IDXEUR_Candlestick_1_M_BID_01.01.2019-31.12.2020.csv')
df3 = pd.read_csv('DEU.IDXEUR_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
DAX = pd.concat([df1, df2, df3])
DAX['Name'] = "DAX"
DAX['Type'] = "Index"

df1 = pd.read_csv('HKG.IDXHKD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('HKG.IDXHKD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
HK = pd.concat([df1, df2])
HK['Name'] = "HK"
HK['Type'] = "Index"

df1 = pd.read_csv('USA500.IDXUSD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('USA500.IDXUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
SP = pd.concat([df1, df2])
SP['Name'] = "S&P"
SP['Type'] = "Index"

df1 = pd.read_csv('USATECH.IDXUSD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('USATECH.IDXUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
NAS = pd.concat([df1, df2])
NAS['Name'] = "NASDAQ"
NAS['Type'] = "Index"

df1 = pd.read_csv('GBR.IDXGBP_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('GBR.IDXGBP_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
FTSE = pd.concat([df1, df2])
FTSE['Name'] = "FTSE"
FTSE['Type'] = "Index"

df1 = pd.read_csv('USTBOND.TRUSD_Candlestick_1_M_BID_18.12.2018-31.12.2020.csv')
df2 = pd.read_csv('USTBOND.TRUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
UST = pd.concat([df1, df2])
UST['Name'] = "UST"
UST['Type'] = "BOND"

df1 = pd.read_csv('EURUSD_Candlestick_1_M_BID_18.12.2018-31.12.2020.csv')
df2 = pd.read_csv('EURUSD_Candlestick_1_M_BID_01.01.2020-30.04.2022.csv')
EURUSD = pd.concat([df1, df2])
EURUSD['Name'] = "EUR/USD"
EURUSD['Type'] = "FOREX"

df1 = pd.read_csv('BTCUSD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('BTCUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
BITC = pd.concat([df1, df2])
BITC['Name'] = "BITCOIN/USD"
BITC['Type'] = "CRYPTO"

df1 = pd.read_csv('ETHUSD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('ETHUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
ETH = pd.concat([df1, df2])
ETH['Name'] = "ETHER/USD"
ETH['Type'] = "CRYPTO"

df1 = pd.read_csv('GAS.CMDUSD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('GAS.CMDUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
GAS = pd.concat([df1, df2])
GAS['Name'] = "GAS"
GAS['Type'] = "COMMODITIES"

df1 = pd.read_csv('BRENT.CMDUSD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('BRENT.CMDUSD_Candlestick_1_M_BID_01.01.2021-30.05.2022.csv')
OIL = pd.concat([df1, df2])
OIL['Name'] = "OIL"
OIL['Type'] = "COMMODITIES"

df1 = pd.read_csv('COFFEE.CMDUSX_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('COFFEE.CMDUSX_Candlestick_1_M_BID_01.01.2020-30.04.2022.csv')
COFFE = pd.concat([df1, df2])
COFFE['Name'] = "COFFE"
COFFE['Type'] = "COMMODITIES"

df1 = pd.read_csv('EEM.USUSD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('EEM.USUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
EEM = pd.concat([df1, df2])
EEM['Name'] = "EMERGINGMARKETS"
EEM['Type'] = "ETF"

df1 = pd.read_csv('IYR.USUSD_Candlestick_1_M_BID_01.01.2018-31.12.2020.csv')
df2 = pd.read_csv('IYR.USUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
IYR = pd.concat([df1, df2])
IYR['Name'] = "REALESTATE"
IYR['Type'] = "ETF"

df1 = pd.read_csv('IWD.USUSD_Candlestick_1_M_BID_01.02.2018-31.12.2020.csv')
df2 = pd.read_csv('IWD.USUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
IWD = pd.concat([df1, df2])
IWD['Name'] = "RUSSEL VALUE"
IWD['Type'] = "ETF"

df1 = pd.read_csv('IWF.USUSD_Candlestick_1_M_BID_01.02.2018-31.12.2020.csv')
df2 = pd.read_csv('IWF.USUSD_Candlestick_1_M_BID_01.01.2021-30.04.2022.csv')
IWF = pd.concat([df1, df2])
IWF['Name'] = "RUSSEL GROWTH"
IWF['Type'] = "ETF"

df = pd.concat([DAX, HK, SP, NAS, FTSE, UST, EURUSD, BITC, ETH, GAS, OIL, COFFE, EEM, IYR, IWD, IWF], ignore_index=True)
CRYPTO = pd.concat([BITC, ETH], ignore_index=True)

df.to_csv('SwissData.txt', index = False, header=True)

#os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")
#date_cols_tmp = ['Local time']
#df1 = pd.read_csv('SwissData.txt', index_col=None, parse_dates=date_cols_tmp)

# 3. Creating functions

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

# 4. Applying functions to data
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

df = pd.concat([DAX, HK, SP, NAS, FTSE], ignore_index=True)
data = timefunc(df, 0)
data['Name'].value_counts()

data.to_csv('IndexDataHour.txt', index = False, header=True)

# 4.5
os.chdir("/Users/frederikzobbe/Desktop/Data")
data = timefunc(df, 0)


# 5. Reading in the data
os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")

def reading(nrows, nskips = 0):
    starttime = time.time()
    df_iter = pd.read_csv('SwissData2.txt', index_col=None, parse_dates=date_cols_tmp, skiprows = nskips, chunksize=nrows)
    df = df_iter.get_chunk()
    
    endtime = time.time()
    dur = endtime - starttime
    print(' --- The function READING took %s minutes to run ---' %round(dur,2)/60)
    return df

#DAX: skiprows=0, nrows = 1394003
def reading(nrows, nskips):
    starttime = time.time()
    date_cols_tmp = ['Local time', 'CET', 'tmptime']
    df_iter = pd.read_csv('SwissData2.txt', index_col=None, parse_dates=date_cols_tmp, skiprows = nskips, chunksize=nrows)
    df = df_iter.get_chunk()
    
    endtime = time.time()
    dur = endtime - starttime
    print(' --- The function READING took %s minutes to run ---' %round(dur,2)/60)
    return df

df = reading(4000000, 0)
df['Type'].value_counts()



df[df['Name'] == 'DAX'].tail(5)

starttime = time.time()
length = 500000
df_iter = pd.read_csv('SwissData2.txt', index_col=None, parse_dates=date_cols_tmp, chunksize=length)
df = df_iter.get_chunk()
len(df)
df['Name'].nunique()
endtime = time.time()


for iter_num, chunk in enumerate(df_iter, 1):
    print(f'Processing iteration {iter_num}')
    # do things with chunk


df = pd.read_csv('SwissData2.txt', sep=',', index_col=None, parse_dates=date_cols_tmp, chunksize=1000) #skiprows=1000)


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





################## DEVELOPMENT ##############
# Dictionary of opening times

Dict = {'DAX': [9,], 1: [1, 2, 3, 4]}
print("\nDictionary with the use of Mixed Keys: ")

# FUNCTION: Determine whether time is preopen, open or postopen
def rowconv(row, openhour, openminute, closinghour, closingminute):
    if (row['Hour'] <= openhour and row['Minute'] < openminute or row['Hour'] < openhour) :
       return 'Early'
    if (row['Hour'] >= openhour and row['Minute'] >= openminute and row['Hour'] <= closinghour and row['Minute'] < closingminute) :
        return 'Open'
    if (row['Hour'] >= closinghour and row['Minute'] >= closingminute or row['Hour'] > closinghour) :
        return 'Late'

DAX['OpenType'] = df_tmp.apply (lambda row: rowconv(row, openhour = 9, openminute = 30, closinghour = 17, closingminute = 30), axis=1)

def openfunc(df: pd.DataFrame):
    if (df['Hour'] <= 9 & df['Minute'] <= 30):
        
    if pd.to_datetime(df.iloc[:,colint]) <= dt.time(9,0):
        print('yes')
    return

df = pd.DataFrame({'coname1': ['Apple','Yahoo','Gap Inc'], 'coname2':['Apple', 'Google', 'Apple']})
df['eq'] = df.apply(lambda row: row['coname1'] == 'Apple', axis=1)

def func(x: pd.DataFrame, colint: int, openhour: dt.time): #, closehour: dt):
    tmpcol    = pd.to_datetime(x.iloc[:,colint])
    x['time'] = tmpcol.dt.time
    
    cond1 = x['time'] <= openhour
    cond2 = x['time'] > openhour
    x[cond1]['time'] = 0
    x[cond2]['time'] = 1
    
    for i in np.arange(x):
        if x.iloc[i,:]['time'] <= openhour:
            print('yes')
        else:
            print('no')
    return x

func(x[:1000], 0, dt.time(9,0))

dt.time(8, 0)

tmp = pd.to_datetime(df[:10000]['Local time'])
tmp.dt.time



type(df['Local time'])

df['Local time']
du_sub = df.sample(1000)

tmp.hour
tmp.dt.time

print(tmp.dt.strftime('%H:%M'))


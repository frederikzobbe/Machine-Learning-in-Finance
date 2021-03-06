# Navn: Reading in data
# Oprettet: 25-05-2022
# Senest ændret: 30-05-2022

################### CHANGELOG ###########################
# FZC: Oprettede programmet                             #              
# FZC: Tilføjede linjer til at indlæse data             #
################### DESCRIPTION #########################
# Programmet indlæser swissdata                         #
#########################################################

# 1. Reading in packages
import pandas as pd
import os as os
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

# 2. Reading in data
os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")
#os.chdir("/Users/mathiasfrederiksen/Desktop/Forsikringsmatematik/5. år/Applied Machine Learning/Data/SwissData")
#os.chdir("/Users/mikki/Desktop/AppML/New env Personal projects/Final project/data/SwissData")

## ------------------------- ValutaData -----------------------------------------
valutadatahour = pd.read_csv('ValutaDataHour_v3.txt', index_col=None, parse_dates=['CET'], engine='python')

## ------------------------ CommoData ---------------------------------------
commodatahour = pd.read_csv('CommoDataHour_v3.txt', index_col=None, parse_dates=['CET'], engine='python')

## ------------------------ IndexData ----------------------------------------
idxdatah = pd.read_csv("IndexDataHour_v3.txt", index_col=None, parse_dates=['CET'], engine='python')

## ------------------------ Subset Data ----------------------------------------
valutadatahour[valutadatahour['Name'] == 'ETHER']
valutadatahour['Name'].value_counts()

## ------------------------ Data read for predictions ----------------------------------------
os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData/predictions")
DAX_8ha     = pd.read_csv("DAX_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
SP_8ha      = pd.read_csv("SP_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
NAS_8ha     = pd.read_csv("NASDAQ_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
HK_8ha      = pd.read_csv("HK_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
FTSE_8ha    = pd.read_csv("FTSE_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
EUR_8ha     = pd.read_csv("EURUSD_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
BIT_8ha     = pd.read_csv("BITCOIN_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
ETH_8ha     = pd.read_csv("ETHER_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
COF_8ha     = pd.read_csv("COFFEE_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
GAS_8ha     = pd.read_csv("GAS_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')
OIL_8ha     = pd.read_csv("OIL_8ha_pred_val_v2.txt", index_col=None, parse_dates=['CET'], engine='python')

datapred = pd.concat([DAX_8ha, SP_8ha, NAS_8ha, HK_8ha, FTSE_8ha, EUR_8ha, BIT_8ha, ETH_8ha, COF_8ha, GAS_8ha, OIL_8ha])
datapred.columns = ['CET', 'Name', 'Close', 'Predictions', 'Diff']

os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData/predictions")
DAX_8ha     = pd.read_csv("DAX_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
SP_8ha      = pd.read_csv("SP_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
NAS_8ha     = pd.read_csv("NAS_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
HK_8ha      = pd.read_csv("HK_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
FTSE_8ha    = pd.read_csv("FTSE_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
EUR_8ha     = pd.read_csv("EURUSD_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
BIT_8ha     = pd.read_csv("BITCOIN_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
ETH_8ha     = pd.read_csv("ETHER_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
COF_8ha     = pd.read_csv("COFFEE_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
GAS_8ha     = pd.read_csv("GAS_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
OIL_8ha     = pd.read_csv("OIL_8ha_pred_lr_v1.txt", index_col=None, parse_dates=['CET'], engine='python')

datapred2 = pd.concat([DAX_8ha, SP_8ha, NAS_8ha, HK_8ha, FTSE_8ha, EUR_8ha, BIT_8ha, ETH_8ha, COF_8ha, GAS_8ha, OIL_8ha])
datapred2.columns = ['CET', 'Name', 'Close', 'Predictions', 'Diff']





# Test from net https://stackoverflow.com/questions/67319653/index-x-axis-as-datetime64ns-not-working
df4 = pd.read_csv('https://pastebin.pl/view/raw/1046cfca',parse_dates=['Open time']) 
df4.set_index('Open time', inplace=True)

fig, (ax2) = plt.subplots(1, 1, figsize=(15, 7))

df4['Close'].plot(ax=ax2, color = 'k', lw = 1, label = 'Close Price')
df4['4_EMA'].plot(ax=ax2, color = 'b', lw = 1, label = '4_EMA')
#df4['20_EMA'].plot(ax=ax2, color = 'g', lw = 1, label = '20_EMA')
        
ax2.set_title("4H time frame")

# ----------------------------------------------------------------------
# the following is changed to use `df.plot`:
# ----------------------------------------------------------------------

# plot 'buy' signals
df4.loc[df4['Position']==1, '4_EMA'].plot(
    ls='None', marker='^', markersize = 15,
    color = 'g', alpha = 0.7, label = 'buy', ax=ax2)

# plot 'sell' signals
df4.loc[df4['Position']==-1, '4_EMA'].plot(
    ls='None', marker='v', markersize = 15,
    color = 'r', alpha = 0.7, label = 'sell', ax=ax2)

plt.show()
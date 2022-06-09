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
#os.chdir("/Users/frederikzobbe/Documents/GitHub/Machine-Learning-in-Finance/Data")
#os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")
#os.chdir("/Users/mathiasfrederiksen/Desktop/Forsikringsmatematik/5. år/Applied Machine Learning/Data/SwissData")
#os.chdir("/Users/mikki/Desktop/AppML/New env Personal projects/Final project/data/SwissData")

date_cols = ['CET']
idxdatah = pd.read_csv('IndexDataHour_v3.txt', index_col=None, parse_dates=date_cols, engine='python')

## ------------------------- DaxData -----------------------------------------
daxdatahour = pd.read_csv('DaxDataHour.txt', index_col=None, parse_dates=['CET'], engine='python')
# Needs fixing. TmpTime still present (07/06)

## ------------------------ CryptoData ---------------------------------------
# Needs fixing. TmpTime still present (07/06)
cryptodata = pd.read_csv('CryptoData.txt', index_col=None, parse_dates=['CET'], engine='python')

## ------------------------ IndexData ----------------------------------------
indexdataHour = pd.read_csv("IndexDataHour.txt", index_col=None, parse_dates=['CET'], engine='python')









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
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
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

pd.options.mode.chained_assignment = None  # default='warn'

# 2. Reading in data
#os.chdir("/Users/frederikzobbe/Documents/GitHub/Machine-Learning-in-Finance/Data")
#os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")

#dtypes = [float, float, float, float, float, str, str, datetime, int, int, int, int, int]
date_cols = ['Local time, tmptime, CET']
cryptodata = pd.read_csv('CryptoData.txt', index_col=None, parse_dates=date_cols)
indexdata = pd.read_csv('IndexData.txt', index_col=None, parse_dates=date_cols)
commodata = pd.read_csv('CommoData.txt', index_col=None, parse_dates=date_cols)

#df.sort_values(by=['Name', 'Year', 'Month', 'Day', 'Hour', 'Minute'], inplace = True, ascending = (1, 1, 1, 1, 1, 1))

# 3. Plotting the data

# old
df_sub = DAX[:100000]
df_sub.set_index('Local time', inplace=True)
plotdata = df_sub["Close"].copy()

plotdata.plot(figsize=(30,7), fontsize = 12)
plt.style.use("seaborn")
plt.show()

# new
df_sub = DAX2[:100000]
df_sub.set_index('Local time', inplace=True)
plotdata = df_sub["Close"].copy()

plotdata.plot(figsize=(30,7), fontsize = 12)
plt.style.use("seaborn")
plt.show()
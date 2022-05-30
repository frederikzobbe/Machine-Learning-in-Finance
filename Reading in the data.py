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

pd.options.mode.chained_assignment = None  # default='warn'

# 2. Reading in data
os.chdir("/Users/frederikzobbe/Documents/GitHub/Machine-Learning-in-Finance/Data")
os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")

date_cols = ['CPHTime']
#dtypes = [float, float, float, float, float, str, str, datetime, int, int, int, int, int]
df = pd.read_csv('SwissData2.txt', index_col=None, parse_dates=date_cols)

# 3. Plotting the data
df_sub = df[df['Name'] == 'DAX']
df_sub.set_index('CPHTime', inplace=True)
plotdata = df_sub["Open"].copy()

plotdata.plot(figsize=(30,7), fontsize = 12)
plt.style.use("seaborn")
plt.show()

plotdata.plot(legend=True,figsize=(12,5))
plt.show()

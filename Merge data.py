
# For this program one can only use data that has been through 
# the timefunc function

# 1. Reading in packages
import matplotlib
import pandas as pd
import numpy as np
import time
import os as os

# 2. Set path for extra data
os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/Anden data")
#os.chdir(r"C:\Users\Bruger\Documents\Google_Drev_data\Anden data")

# 3. Read in data
df_SB       = pd.read_csv('Storebalt_month.csv', sep=';')
df_FB_index = pd.read_csv('ForbrugerIndex.csv', sep=';')
df_PE       = pd.read_csv('PE.csv', sep=';')

# 4. Set path for primary data
os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")
#os.chdir(r"C:\Users\Bruger\Documents\Google_Drev_data\SwissData")

# 5. Read in primary data
df = pd.read_csv('CommoDataHour.txt', index_col=None, parse_dates=['CET'], engine='python')

# 6. Define functions
def ROC(df, unit):
    if isinstance(df, pd.DataFrame):
        final_df = pd.DataFrame()
        for name in pd.unique(df['Name']):
            
            df_type = df[df['Name'] == name]
            
            close = df_type['Close']
            ROC_df = close.pct_change(periods=unit)
            
            final_df = pd.concat([final_df,ROC_df])
            
        final_df.replace(np.nan, 0, inplace = True)       
        return final_df.to_numpy().reshape(-1)
        
    else:
        print('Input need to be a Pandas DataFrame')
        
def EMA(df, days, smoothing = 2):
    if isinstance(df, pd.DataFrame):
        final_list = []
        k = smoothing / (days + 1)
        
        for name in pd.unique(df['Name']):
            
            df_type = df[df['Name'] == name]
            df_type = df_type.reset_index(drop=True)
            
            EMA_list = [0]*days
            SMA = np.mean(df_type['Close'][0:days])
            EMA = SMA # Our first previous EMA
            
            for n in np.arange(days,df_type.shape[0]):
                
                close = df_type['Close'][n]
                EMA = close * k + EMA * (1-k)
                EMA_list.append(EMA)
                
                if n % 100000 == 0:
                    print(n)
            
            final_list = final_list + EMA_list
        
        return final_list
        
    else:
        print('Input need to be a Pandas DataFrame')

# 7. Calculate (and concatenate) values
ROC_5 = ROC(df, 5)
ROC_10 = ROC(df, 10)
ROC_15 = ROC(df, 15)
ROC_20 = ROC(df, 20)        

EMA_10 = EMA(df, 5)
EMA_50 = EMA(df, 10)
EMA_200 = EMA(df, 15)

df_dummy1 = pd.DataFrame({'ROC-5' : ROC_5, 'ROC-10' : ROC_10, 'ROC-15' : ROC_15, 'ROC-20' : ROC_20})
df_dummy2 = pd.DataFrame({'EMA-5' : EMA_10, 'EMA-10' : EMA_50, 'EMA-15' : EMA_200})
df = pd.concat([df,df_dummy1, df_dummy2], axis = 1)

# 8. Merge Storebæltsbrodata and forbrugerprisindeks på data
merged_df = df.merge(df_SB, how = 'inner', on = ['Year', 'Month'])
merged_df = merged_df.merge(df_FB_index, how = 'inner', on = ['Year', 'Month'])
#merged_df1 = merged_df.merge(df_PE, how = 'inner', on = ['Year', 'Month', 'Name']) # Only for indexdata

merged_df.to_csv('CommoDataHour_v3.txt', index = False, header=True)
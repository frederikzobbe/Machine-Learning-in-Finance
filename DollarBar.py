import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os as os
import math


#os.chdir(r"C:\Users\Bruger\Documents\Google_Drev_data\SwissData")
date_cols = ['CET']
df = pd.read_csv('20MinIndexData', index_col=None, parse_dates=['CET'], engine='python')
df_sub = df.iloc[0:2000000,:]





def roundup(x, k):
    return int(math.ceil(x / k)) * k

Value = []
for name in pd.unique(df['Name']):
    x = np.max(df[df['Name'] == name]['High'])*np.max(df[df['Name'] == name]['Volume'])
    print(x)
    Value.append(len(str(int(x/100))))

roundup_df = pd.DataFrame({'Name' : pd.unique(df['Name']), 'Value' : Value})



def dollarbar(df, time_unit, roundup_df):
    if isinstance(df, pd.DataFrame):

        ## Create empty dataframe
        df_Dollarbar = pd.DataFrame({'CPHTime': [], 'Period': [], 'Open': [], 'Low': [], 'High': [],
                                     'Close': [], 'Volume': [], 'Monetary_Volume': [], 'Name': [], 'Type': []})

        for name in pd.unique(df['Name']):

            df_type = df[df['Name'] == name]
            df_type = df_type.reset_index(drop=True)
            Rest_transaction = 0
            Total_transaction = 0
            Type = df_type['Type'][0]

            ## Empty lists
            Time_list = []
            Year_list = []
            Month_list = []
            Day_list = []
            Open_list = []
            Low_list = []
            High_list = []
            Close_list = []
            Volume_list = []
            HL_list = []
            Period_list = []

            ## support vairables
            Volume_help_list = []
            iter_since_dollarbar = 0

            ## Dollarsbar cap
            max_High = np.max(df_type['High'])
            max_Volume = np.max(df_type['Volume'])
            Dollarbar_cap = roundup(max_High * max_Volume,
                                    10 ** roundup_df[roundup_df['Name'] == 'DAX']['Value'].item())

            ## First value in Open
            Open_list.append(df_type['Open'][0])

            for n in np.arange(df_type.shape[0]):

                Time = df_type['CET'][n]
                Year = df_type['Year'][n]
                Month = df_type['Month'][n]
                Day = df_type['Day'][n]
                Open = df_type['Open'][n]
                High = df_type['High'][n]
                Low = df_type['Low'][n]
                Close = df_type['Close'][n]
                Volume = df_type['Volume'][n]

                Mean_HL = np.mean([High, Low])
                HL_list.append(High)
                HL_list.append(Low)

                Total_transaction = Total_transaction + Mean_HL * Volume
                Mean_OC = np.mean([Open, Close])

                Volume_help_list.append(Volume)

                iter_since_dollarbar += 1

                if n % 10000 == 0:
                    print(n)

                if Dollarbar_cap < Total_transaction:  # and df_type.shape[0] > n

                    Rest_transaction = Total_transaction - Dollarbar_cap

                    max_High = np.max(HL_list)
                    min_Low = np.min(HL_list)

                    p = (Dollarbar_cap - Total_transaction + Mean_HL * Volume) / (
                                Mean_HL * Volume)  ## andel af volume som skal med i denne dollarsbar
                    Volume_help_list[-1] = Volume * p
                    Sum_volume = np.sum(Volume_help_list)

                    ## Saving dollarbar values
                    Time_list.append(Time)
                    Year_list.append(Year)
                    Month_list.append(Month)
                    Day_list.append(Day)
                    Period_list.append(iter_since_dollarbar * time_unit)
                    Low_list.append(min_Low)
                    High_list.append(max_High)
                    Close_list.append(Mean_OC)
                    Volume_list.append(Sum_volume)
                    Open_list.append(Mean_OC)

                    HL_list = []

                    Volume_help_list = []
                    Volume_help_list.append(Volume * (1 - p))  ## andel af volume som skal med i næste dollarsbar

                    Total_transaction = Rest_transaction

                    iter_since_dollarbar = 0

                if df_type.shape[0] == n + 1 and iter_since_dollarbar == 0:
                    Sum_volume = Volume * (1 - p)

                    Close_list.append(Close)
                    Time_list.append(Time)  ## Time bliver det samme, hvilket ik er godt
                    Year_list.append(Year)
                    Month_list.append(Month)
                    Day_list.append(Day)
                    Low_list.append(min_Low)
                    High_list.append(max_High)
                    Volume_list.append(Sum_volume)

                    df_dummy = pd.DataFrame({'CET': Time_list, 'Year': Year_list, 'Month': Month_list, 'Day': Day_list,
                                             'Period': Period_list, 'Open': Open_list, 'Low': Low_list,
                                             'High': High_list,
                                             'Close': Close_list, 'Volume': Volume_list,
                                             'Monetary_Volume': Dollarbar_cap, 'Name': name, 'Type': Type})
                    df_Dollarbar = pd.concat([df_Dollarbar, df_dummy.iloc[:-1, :]])

                if df_type.shape[0] == n + 1 and iter_since_dollarbar > 0:
                    max_High = np.max(HL_list)
                    min_Low = np.min(HL_list)

                    Sum_volume = np.sum(Volume_help_list)

                    ## Saving dollarbar values
                    Time_list.append(Time)
                    Year_list.append(Year)
                    Month_list.append(Month)
                    Day_list.append(Day)
                    Period_list.append(iter_since_dollarbar * time_unit)
                    Low_list.append(min_Low)
                    High_list.append(max_High)
                    Close_list.append(Mean_HL)
                    Volume_list.append(Sum_volume)

                    df_dummy = pd.DataFrame({'CET': Time_list, 'Year': Year_list, 'Month': Month_list, 'Day': Day_list,
                                             'Period': Period_list, 'Open': Open_list, 'Low': Low_list,
                                             'High': High_list,
                                             'Close': Close_list, 'Volume': Volume_list,
                                             'Monetary_Volume': Dollarbar_cap, 'Name': name, 'Type': Type})
                    df_dummy['Monetary_Volume'].iloc[-1] = Total_transaction
                    df_Dollarbar = pd.concat([df_Dollarbar, df_dummy])

        return df_Dollarbar

    else:
        print('Input need to be a Pandas DataFrame')


## -------------------------------- Creating dollar bars for hourly and 20 min index data ------------------------
IndexDolla = dollarbar(df, 1, roundup_df)

IndexDolla.dtypes
IndexDolla = IndexDolla.astype({"Year": 'int', "Month": 'int', "Day": 'int', "Period": 'int'})
IndexDolla = IndexDolla.drop(["CPHTime"], axis=1)

os.getcwd()
os.chdir("/Users/mikki/Desktop/AppML/New env Personal projects/Final project/data/SwissData")
IndexDolla.to_csv("IndexDataHourly_DB", index=None, index_label=False, header=True)

## ----------------- 20 min Index data ------------------------
Index20Min_DB = dollarbar(df, 1, roundup_df)

Index20Min_DB = Index20Min_DB.astype({"Year": 'int', "Month": 'int', "Day": 'int', "Period": 'int'})
Index20Min_DB = Index20Min_DB.drop(["CPHTime"], axis=1)
Index20Min_DB = Index20Min_DB.reset_index(drop=True)

os.getcwd()
os.chdir("/Users/mikki/Desktop/AppML/New env Personal projects/Final project/data/SwissData")
Index20Min_DB.to_csv("IndexData20Min_DB", index=None, index_label=False, header=True)

# Check everything is as it should be
FTSEidx = Index20Min_DB.Name == "FTSE"
plt.plot(Index20Min_DB[FTSEidx].CET, Index20Min_DB[FTSEidx].Close )
plt.close()

# Looks like there is a one-timestep ridicolous drop at corona. So we check returns
log_returns = np.log(Index20Min_DB[FTSEidx].Close/Index20Min_DB[FTSEidx].Close.shift(1)).dropna()
log_returns = log_returns.reset_index(drop=True)
plt.hist(log_returns, np.arange(min(log_returns), max(log_returns), 0.001))
plt.close()

# largest drop is -0.13. So we check how large a drop this is
log_returns[log_returns < -0.1]
hmm = Index20Min_DB[FTSEidx].reset_index(drop=True)  # See from hmm that largest drop is approx 15% which is large but acceptable

## ------------------ Alternative dollarbar (varying sizes) -----------------------------

def bar(x,y):
    return np.int64(x/y)*y


df = IndexDat20Min # From reading in the data
name = 'FTSE'
def VaryingDollarbar(df):
    if isinstance(df, pd.DataFrame):
        ## Create empty dataframe
        df_DollarbarVar = pd.DataFrame({'Open': [], 'High': [], 'Low': [], 'Close': [], 'Volume': [],
                                     'Name': [], 'Type': [], 'CET': [], 'Year': [], 'Month': [], 'Day': [],
                                     'Price': [], 'helper': [], 'groupIdx': []})

        df['Price'] = ((df.Open + df.Close)*df.Volume)/2 # Need price to know when to split bars.

        for name in pd.unique(df['Name']):
            df_temp = df[df.Name == name].reset_index(drop=True)

            mask = df_temp.Price != 0
            market_value = np.int64(np.mean(df_temp[mask].Price) + np.std(df_temp[mask].Price))  # Average traded price for observations where trades have happend
            # We want to group up till market_value has been traded. We create helper column to do this
            helper = []
            price = 0
            for i in np.arange(len(df_temp)):
                if price < market_value:
                    price += df_temp.Price[i]
                    helper.append(price)
                else:
                    price = df_temp.Price[i]
                    helper.append(price)

            df_temp['helper'] = helper


            # Now we use the helper column to create an index for the desired groupings

            groupIdx = []
            Idx = 0
            for i in np.arange(len(df_temp)):
                if df_temp.helper[i] < market_value:
                    if df_temp.helper.shift(1).fillna(0)[i] >= market_value:
                        Idx += 1
                        groupIdx.append(Idx)
                    else:
                        groupIdx.append(Idx)
                if df_temp.helper[i] >= market_value:
                    if df_temp.helper.shift(1).fillna(0)[i] >= market_value:
                        Idx += 1
                        groupIdx.append(Idx)
                    else:
                        groupIdx.append(Idx)

            df_temp["groupIdx"] = groupIdx

            dollar_bars_temp = df_temp.groupby(np.array(df_temp.groupIdx)).agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum',
                'Name': 'first', 'Type': 'first', 'CET': 'first', 'Year': 'first', 'Month': 'first', 'Day': 'first',
                'Price': 'sum', 'helper': 'first', 'groupIdx': 'first'
            })

            df_DollarbarVar = pd.concat([df_DollarbarVar, dollar_bars_temp])

        return df_DollarbarVar.drop(['helper', 'groupIdx'], axis=1)

Index_20Min_varyingDB = VaryingDollarbar(df)

maskTest = Index_20Min_varyingDB.Name == "DAX"
Index_20Min_varyingDB['test'] = Index_20Min_varyingDB[maskTest].Price - 44509101111
testIdx = Index_20Min_varyingDB.test < 0
Index_20Min_varyingDB.Price[testIdx] # Only last dollarbar is negative. so its fine

Index_20Min_varyingDB = Index_20Min_varyingDB.drop(['test'], axis = 1)
# Save to data
os.getcwd()
Index_20Min_varyingDB.to_csv("index_20min_varyingDB", index=None, index_label=False)

test = pd.read_csv('index_20min_varyingDB')

## --------------- Code to understand how VaryingDollar function works.
tester = pd.DataFrame({'col1': [1, 1, 1, 3, 4, 1, 1, 1, 1, 1], "col2": [1,2,3,4,5,6,7,8,9,10]})
test = 3
len(bar(np.cumsum(tester.col1.shift(1).fillna(0)),test))
len(np.array(tester.groupIdx))
tester['cumsum'] = np.cumsum(tester.col1.shift(1).fillna(0))
tester['bar'] = bar(tester['cumsum'],test)
tester
cumsum2 = []
price = 0
i = 3
for i in np.arange(len(tester)):
    if price < 3:
        price += tester.col1[i]
        cumsum2.append(price)
    else:
        price = tester.col1[i]
        cumsum2.append(price)


tester['cumsum2'] = cumsum2

groupIdx = []
Idx = 0
            for i in np.arange(len(tester)):
                if tester.cumsum2[i] < test:
                    if tester.cumsum2.shift(1).fillna(0)[i] >= test:
                        Idx += 1
                        groupIdx.append(Idx)
                    else:
                        groupIdx.append(Idx)
                if tester.cumsum2[i] >= test:
                    if tester.cumsum2.shift(1).fillna(0)[i] >= test:
                        Idx += 1
                        groupIdx.append(Idx)
                    else:
                        groupIdx.append(Idx)

tester['groupIdx'] = groupIdx
np.array(tester.groupIdx)

whaaat = tester.groupby(np.array(tester.groupIdx)).agg({'col1':'sum', 'col2':'sum', 'cumsum': 'first',
                                                        'bar': 'first', 'cumsum2': 'sum', 'groupIdx':'first'})
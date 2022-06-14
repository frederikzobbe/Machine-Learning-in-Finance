# 1. Reading in packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time as time

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#013c9b",# my Blue
                                                    "#f6960a",# my Orange
                                                    "#e0012e",# my Red
                                                    "#80cd59",# my Green
                                                    "#4f94d4",# my Light Blue
                                                    "#ffd800",# my Yellow
                                                    "#6f7072",# my Grey
                                                    ])

def hour_to_day(x: pd.DataFrame):

    starttime = time.time()
    
    x.reset_index(inplace=True, drop=True)

    x['Hour']   = x['CET'].dt.hour
    x['Date']   = [str(date)[:10] for date in x['CET']]

    names  = x['Name'].unique()
    dates = x['Date']

    alldates = np.unique([str(date) for date in dates])

    for name in names:
        date = x[x['Name'] == name]['Date']
        tmp   = np.unique([str(date) for date in date])
        alldates  = set(alldates).intersection(set(tmp))

    idxs_all = []
    for d in alldates:
        array = []
        idxs_day = x[(x['Date'] == d)].index

        for name in names:
            tmparray = x[(x['Name'] == name) & (x['Date'] == d)]['Hour']
            array.append(tmparray.values)

        matrix = pd.DataFrame(index=range(len(array[0])),columns=range(len(names)-1))
        for i in np.arange(len(names)-1):
            for j, value in enumerate(array[0]):
                matrix.iloc[j,i] = min(np.abs(value - array[i+1]))
    
        sumtrix = matrix.sum(axis=1)
        element = sumtrix.index[sumtrix == min(sumtrix)][0]

        idxs_all.append(idxs_day[element])

        for i in np.arange(len(names)-1):
            dist    = min(np.abs(array[i+1]-array[0][element]))
            tmptrix = np.abs(array[i+1]-array[0][element])
            place   = np.where(tmptrix == dist)[0][0]
            
            idxs_all.append( (x[(x['Name'] == names[i+1]) & (x['Date'] == d)].index)[place].item() ) 
    
    Close_8hb = pd.DataFrame()
    for name in names:
        tmp = x[x['Name'] == name]['Close'].shift(1).fillna(0)
        Close_8hb = pd.concat([Close_8hb, tmp])
    
    x['Close_8hb'] = Close_8hb

    x.drop("Hour", axis=1, inplace=True)
    
    # Ends the timer
    endtime = time.time()
    dur = endtime - starttime
    print(' --- The function hour_to_day took %s seconds to run ---' %round(dur,2))

    return x.iloc[idxs_all,:].sort_values(by=['CET']).reset_index(drop=True)

def afkast(data, begin_CET, pred_true, date_cet = 'CET'): #predict data
    
    stock_returns = pd.DataFrame()
    
    df_uniq = pd.DataFrame({date_cet : np.sort(pd.unique(data[date_cet]), axis = 0)})
    for name in pd.unique(data['Name']):
        uniq = pd.unique(data[data['Name'] == name][date_cet])
        uniq = pd.DataFrame({date_cet : uniq})
        df_uniq = df_uniq.merge(uniq, how = 'inner', on = [date_cet])
    
    if(pred_true == 'TRUE' or pred_true == 'true' or pred_true == 'True'):
    
        for name in pd.unique(data['Name']):
            df_name = data[data['Name'] == name].reset_index(drop = True)
            df_sub = df_name.merge(df_uniq, how = 'inner', on = [date_cet])
            idx = df_sub[df_sub[date_cet] == begin_CET].index[0]
            df = df_sub.iloc[idx-1:,:].reset_index(drop = True)
            
            ret = []
            for i in np.arange(len(df)-1):
                ret.append((df['Predictions'][i+1]- df['Close_8hb'][i+1])/df['Close_8hb'][i+1])
            
            stock_returns[name] = np.array(ret)
    
        stock_returns = stock_returns.dropna() # drop the first row.'
        
    else:
        for name in pd.unique(data['Name']):
            df_name = data[data['Name'] == name].reset_index(drop = True)
            df_sub = df_name.merge(df_uniq, how = 'inner', on = [date_cet])
            idx = df_sub[df_sub[date_cet] == begin_CET].index[0]
            df = df_sub.iloc[idx-1:,:].reset_index(drop = True)
            
            ret = []
            for i in np.arange(len(df)-2):
                ret.append((df['Close_8hb'][i+2]- df['Close_8hb'][i+1])/df['Close_8hb'][i+1])
            
            stock_returns[name] = np.array(ret)
    
        stock_returns = stock_returns.dropna() # drop the first row.'
        
    return stock_returns

# Historical volatility
def volatility(data, begin_CET, pred_true, date_cet = 'CET'): #predict data, need CET, Name, Close, pred
    
    df_vol = pd.DataFrame()
    
    #date = begin_CET
    #year = date.dt.year.values[0]
    #new_year = str(year-1)
    #date = new_year + date[4:]
    
    df_uniq = pd.DataFrame({date_cet : np.sort(pd.unique(data[date_cet]), axis = 0)})
    for name in pd.unique(data['Name']):
        uniq = pd.unique(data[data['Name'] == name][date_cet])
        uniq = pd.DataFrame({date_cet : uniq})
        df_uniq = df_uniq.merge(uniq, how = 'inner', on = [date_cet])
    
    if(pred_true == 'TRUE' or pred_true == 'true' or pred_true == 'True'):
        annual_business_days=200        
        for name in pd.unique(data['Name']):
            df_name = data[data['Name'] == name].reset_index(drop=True)
            df_sub = df_name.merge(df_uniq, how = 'inner', on = [date_cet])
            NAN = df_sub['Predictions'].isnull()
            idx = 0 if (sum(NAN == True) == 0) else [i for i, x in enumerate(NAN) if x][-1] 
            #idx = df_name[df_name[date_cet] == date].index.tolist()
            df = df_sub.iloc[(idx-annual_business_days):,:].reset_index(drop=True)
            vol = []
        
            for n in np.arange(df.shape[0]-annual_business_days-1):
                df_close = df['Close'][n+1:n+1+annual_business_days]  # if we want vol for 01/01-2022, then we want to start at 02/01-2021
                df_pred = df['Predictions'][n+annual_business_days+1]
                df_merge = np.concatenate([df_close, pd.Series(df_pred)])
                vol.append((df_merge.std() / np.sqrt(annual_business_days))/df_merge.mean())
        
            df_vol[name] = np.array(vol)
            
    else:
        annual_business_days = 200
        for name in pd.unique(data['Name']):
            df_name = data[data['Name'] == name].reset_index(drop=True)
            df_sub = df_name.merge(df_uniq, how = 'inner', on = [date_cet])
            NAN = df_sub['Predictions'].isnull()
            idx = [i for i, x in enumerate(NAN) if x][-1]
            #idx = df_name[df_name[date_cet] == date].index.tolist()
            df = df_sub.iloc[idx-annual_business_days:,:].reset_index(drop=True)
    
            vol = []
            
        
            for n in np.arange(df.shape[0]-annual_business_days-1):
                df_close = df[n+1:n+annual_business_days+1]  # if we want vol for 01/01-2022, then we want to start at 02/01-2021
                vol.append((df_close['Close'].std() / np.sqrt(annual_business_days))/df_close['Close'].mean())
        
            df_vol[name] = np.array(vol)
        
        
        
    return df_vol

def portfolio(data, numAssets, numRev, data_vol, vol_penalty_factor): ## Afkast data og Vol data
    
    df = data.copy()
    assets_used_daliy = pd.DataFrame()
    selected_assets = []
    avg_daily_ret = [0]
    
    for i in range(len(df)):
        if len(selected_assets ) > 0:
            avg_daily_ret.append(df[selected_assets].iloc[i,:].mean()) # Assumes we invest equally in the selected assets
            bad_assets = df[selected_assets].iloc[i,:].sort_values(ascending=True)[:numRev].index.values.tolist()
            selected_assets  = [t for t in selected_assets if t not in bad_assets]
            
        fill = numAssets - len(selected_assets)
        
        vol = data_vol.iloc[i,:]
        #vol_norm = (vol - min(vol)) / (max(vol) - min(vol))  # values between 0 and 1. If used, use below as return_vol
        return_vol = df.iloc[i,:]-vol_penalty_factor*vol
        
        #return_vol = df.iloc[i,:]/(vol)
        return_vol.drop(selected_assets, axis=0, inplace=True)
        new_picks = return_vol.sort_values(ascending=False)[:fill].index.values.tolist()
        selected_assets  = selected_assets  + new_picks
        print(selected_assets)
        assets_used_daliy[i] = selected_assets
        
    returns_df = pd.DataFrame(np.array(avg_daily_ret),columns=["daily_returns"])
    return returns_df, assets_used_daliy.T 

def cagr(data):  # portfolio data
    df = data.copy()
    df['cumulative_returns'] = (1 + df['daily_returns']).cumprod()
    trading_days = annual_business_days
    n = len(df)/ trading_days
    cagr = (df['cumulative_returns'][len(df)-1])**(1/n) - 1
    return cagr

def maximum_drawdown(data):
    df = data.copy()
    df['cumulative_returns'] =  (1 + df['daily_returns']).cumprod()
    df['cumulative_max'] = df['cumulative_returns'].cummax()
    df['drawdown'] = df['cumulative_max'] - df['cumulative_returns']
    df['drawdown_pct'] = df['drawdown'] / df['cumulative_max']
    max_dd = df['drawdown_pct'].max()
    return max_dd

def calmar_ratio(data, rf):
    df = data.copy()
    calmar = (cagr(df) - rf) / maximum_drawdown(df)
    return calmar

# def volatility2(data):
#     df = data.copy()
#     trading_days = 200
#     vol = df['monthly_returns'].std() * np.sqrt(trading_days)
#     return vol

# def sharpe_ratio(data, rf):
#     df = data.copy()
#     sharpe = (cagr(df) - rf)/ volatility(df)
#     return sharpe

annual_business_days = 200
numAssets=5
numRev=2
vol_penalty_factor=0#.5
datapred3 = datapred.copy()
datapred3['Predictions'] = datapred3[datapred3['Diff'].notnull()]['Close']

# Load data to day-data (re-balancing once a day)
pred_data = hour_to_day(datapred2.copy())

## Pred, the return our prediction say we would get
begin_date = '2021-04-01'
returns_pred = afkast(data=pred_data, begin_CET=begin_date, pred_true='TRUE', date_cet='Date')
vol_pred = volatility(data=pred_data, begin_CET=begin_date, pred_true='TRUE', date_cet='Date')
rebalanced_portfolio_pred, assets_list = portfolio(data=returns_pred, numAssets=numAssets, numRev=numRev, data_vol=vol_pred, vol_penalty_factor=vol_penalty_factor)

## How we really should invest, if we followed our strategy
returns_true = afkast(data=pred_data, begin_CET=begin_date, pred_true='FALSE', date_cet='Date')
vol_true = volatility(data=pred_data, begin_CET=begin_date, pred_true='FALSE', date_cet='Date')
#rebalanced_portfolio_true, _  = Portfolio(returns_true, 3, 3, vol_true, 0.5)
benchmark = pd.DataFrame()
benchmark['daily_returns'] = np.array(returns_true.mean(axis=1))

## The return we really get from Pred
real_avg_daily_ret_from_pred_result = [0]
for i in np.arange(returns_true.shape[0]):
    real_avg_daily_ret_from_pred_result.append(returns_true[assets_list.iloc[i,:]].iloc[i,:].mean()) #probably not good
real_df = pd.DataFrame(np.array(real_avg_daily_ret_from_pred_result),columns=["daily_returns"])   

print("Rebalanced Portfolio Performance - How we hope (predicted) it went")
print("CAGR: " + str(cagr(rebalanced_portfolio_pred)))
print("Maximum Drawdown: " + str(maximum_drawdown(rebalanced_portfolio_pred) ))
print("Calmar Ratio: " + str(calmar_ratio(rebalanced_portfolio_pred, 0.03)))

print("\n")

print("Rebalanced Portfolio Performance - How it really went")
print("CAGR: " + str(cagr(real_df)))
print("Maximum Drawdown: " + str(maximum_drawdown(real_df) ))
print("Calmar Ratio: " + str(calmar_ratio(real_df, 0.03)))

print("\n")

print("Benchmark")
print("CAGR: " + str(cagr(benchmark)))
print("Maximum Drawdown: " + str(maximum_drawdown(benchmark) ))
print("Calmar Ratio: " + str(calmar_ratio(benchmark, 0.03)))





os.chdir("/Users/frederikzobbe/Documents/Universitet/Forsikringsmatematik/Applied Machine Learning/Final project/Final project data/SwissData")
MWL_v1     = pd.read_csv("MWL_v1.txt", index_col=None, parse_dates=['CET'], engine='python')
MWL_v1['Close'].pct_change()

MWL_v1['Predictions'] = MWL_v1['Close']
MWL_v1['Name'] = 'MSCI World Index'


pred_data.head(20)

dates_between1 = pred_data['CET']
dates_between1 = np.unique([str(date)[:10] for date in dates_between1])
MWL_v1 = MWL_v1[MWL_v1['CET'].isin(dates_between1)]

dates_between2 = MWL_v1['CET']
dates_between2 = np.unique([str(date)[:10] for date in dates_between2])
MWL_v1 = MWL_v1[MWL_v1['CET'].isin(dates_between2)]

MSCI = afkast(data=MWL_v1, begin_CET=begin_date, pred_true='TRUE', date_cet='CET')

(1+returns_true).cumprod()
pd.DataFrame(np.array(assets_list).reshape(assets_list.shape[0]*assets_list.shape[1])).value_counts()
pred_data[(pred_data['Name'] == 'COFFEE')][790:820]
returns_pred

set(dates_between1) - set(dates_between2) 
(1+MSCI).cumprod()


#lrcumprod = real_df # logreturn 
#pricecumprod = real_df
#powportfolio =real_df
#fig, ax = plt.subplots(figsize=(8,5), dpi=500)
fig, ax = plt.subplots(figsize=(8,5), dpi=130)
#plt.plot((1+rebalanced_portfolio_pred).cumprod())
#plt.plot((1+powportfolio).cumprod(), label = "Real Strategy Return: Best asset")
plt.plot((1+pricecumprod).cumprod(), label = "Real Strategy Return: Price model")
plt.plot((1+lrcumprod).cumprod(), label = 'Real Strategy Return: Logreturn model')
plt.plot((1+benchmark).cumprod(), label = 'Benchmark Return')
plt.plot((1+MSCI).cumprod(), label = 'MSCI World Index')
plt.title("Benchmark vs Rebalancing Strategy Return")
plt.ylabel("Cumulative return")
plt.xlabel("Business Days")
plt.axhline(y=1, ls='--', c='grey')
plt.axhline(y=1.5, ls='--', c='grey')
ax.legend(loc='upper left')
#ax.legend(["Strategy Return", "Real Strategy Return", "Benchmark Return"])
#plt.show()
plt.savefig('Benchmark vs Rebalancing Strategy Return.png')





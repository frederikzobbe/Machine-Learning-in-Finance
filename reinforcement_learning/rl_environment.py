import pandas as pd
import numpy as np

# import sys
# sys.path.append('..')

def read_and_prepare_data(sec_code,columns):
    prices = pd.read_csv('../jpx_data/train_files/stock_prices.csv')
    prices.Date = pd.to_datetime(prices['Date'])
    prices.set_index('Date',inplace=True)
    prices = prices.loc[prices['SecuritiesCode']==sec_code][columns]
    if prices.isnull().values.any():
        bef = prices.shape[0]
        prices=prices.dropna()
        aft = prices.shape[0]
        print(f'Nan found in dataset, removed rows {bef-aft}')
    return prices

class Rl_env():
    def __init__(self,sec_code,scaler,period_length=10,columns=['Open','Close','Volume','Low','High']):
        self.full_dataset=read_and_prepare_data(sec_code,columns)
        self.index=self.full_dataset.index
        print(f'loaded {self.full_dataset.shape[0]} rows and {self.full_dataset.shape[1]} columns from code {sec_code}')
        print(f'loaded columns:{columns}')

        self.scaler = scaler
        self.period_length = period_length

        self.reset_state()

        self.eps = np.finfo(np.float32).eps.item() 

        

    def current_index(self,):
        return self.full_dataset.iloc[[self.period_start+self.period_length]].index

        
    def get_next_day(self,scale=True):
        row = self.full_dataset.iloc[[self.period_start+self.period_length]]
        if scale:
            return self.scale_data(row,new_fit=False)
        else:
            return row


    def step_period(self,):
        self.period_start+=1
        self.state = self.full_dataset.iloc[self.period_start:self.period_start+self.period_length]
        
        self.state = self.scale_data(self.state,new_fit=True)
        
        
    def scale_data(self,data,new_fit=False,inverse_transform=False):
        indexes = data.index
        columns = data.columns

        if new_fit:
            scaled_data = self.scaler.fit_transform(data.to_numpy())
        else:
            if inverse_transform:
                scaled_data = self.scaler.inverse_transform(data.to_numpy())
            else:
                scaled_data = self.scaler.transform(data.to_numpy())
        
        return pd.DataFrame(data = scaled_data,
                            index = indexes,
                            columns = columns)


    def get_rel_change(self,prediction_variable='Close'):
        next_day = self.get_next_day(scale=False)
        last_day_of_state = self.scale_data(self.state,inverse_transform=True)[prediction_variable].iloc[-1]
        if last_day_of_state == 0:
            relative_change = next_day[prediction_variable].iloc[0]/self.eps 

        relative_change = (next_day[prediction_variable].iloc[0]-last_day_of_state)/last_day_of_state
        return relative_change

    def reset_state(self,):
        self.state = self.full_dataset.iloc[:self.period_length]
        self.state = self.scale_data(self.state,new_fit=True)

        self.period_start=0
        

    def get_reward(self,action):
        if action == 0: #do nothing
            return 0
        rel_change = self.get_rel_change()
        if action == 1: #buy
            return rel_change
        if action == 2: #sell
            return -rel_change

        
    


        
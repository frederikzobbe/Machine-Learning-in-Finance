import pandas as pd
import numpy as np

# import sys
# sys.path.append('..')

def read_and_prepare_data(sec_code,columns):
    prices = pd.read_csv('../jpx_data/train_files/stock_prices.csv')
    prices.Date = pd.to_datetime(prices['Date'])
    prices.set_index('Date',inplace=True)
    return prices.loc[prices['SecuritiesCode']==sec_code][columns]

class Rl_env():
    def __init__(self,sec_code,scaler,period_length=10,columns=['Open','High','Low','Close','Volume']):
        self.full_dataset=read_and_prepare_data(sec_code,columns)
        self.index=self.full_dataset.index
        print(f'loaded {self.full_dataset.shape[0]} rows and {self.full_dataset.shape[1]} columns from code {sec_code}')
        print(f'loaded columns:{columns}')

        self.scaler = scaler

        self.state = self.full_dataset.iloc[:period_length]
        self.state = self.scale_data(self.state,new_fit=True)

        self.period_start=0
        self.period_length = period_length

    def current_index(self,):
        return self.full_dataset.iloc[[self.period_start+self.period_length]].index

        
    def get_next_day(self,):
        row = self.full_dataset.iloc[[self.period_start+self.period_length]]
        return self.scale_data(row,new_fit=False)


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

        
    


        
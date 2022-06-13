import pandas as pd
import numpy as np

# import sys
# sys.path.append('..')
all_prices = pd.read_csv('../jpx_data/train_files/stock_prices.csv')
all_prices.Date = pd.to_datetime(all_prices['Date'])
all_prices.set_index('Date',inplace=True)

def read_and_prepare_data(sec_codes,columns):
    if isinstance(sec_codes,(list,tuple,np.ndarray)):
        prices = all_prices.loc[all_prices['SecuritiesCode'].isin(sec_codes)][columns+['SecuritiesCode']]
        new_prices = pd.DataFrame(index=prices.index.unique(),columns=sec_codes)
        for sec_code in sec_codes:
            new_prices[sec_code] = prices.loc[prices['SecuritiesCode'] == sec_code][columns]
        prices = new_prices
        
    elif isinstance(sec_codes,int):
        prices = all_prices.loc[all_prices['SecuritiesCode']==sec_codes][columns]
        
    else:
        raise Exception('both isinstances false')
    
    if prices.isnull().values.any():
        bef = prices.shape[0]
        prices=prices.dropna()
        aft = prices.shape[0]
        print(f'Nan found in dataset for sec code {sec_codes}, removed {bef-aft} rows')
    return prices


def scale_data(scaler,data,new_fit=False,inverse_transform=False):
    indexes = data.index
    columns = data.columns

    if new_fit:
        scaled_data = scaler.fit_transform(data.to_numpy())
    else:
        if inverse_transform:
            scaled_data = scaler.inverse_transform(data.to_numpy())
        else:
            scaled_data = scaler.transform(data.to_numpy())
    
    return pd.DataFrame(data = scaled_data,
                        index = indexes,
                        columns = columns)

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
            return scale_data(self.scaler,row,new_fit=False)
        else:
            return row


    def step_period(self,):
        self.period_start+=1
        self.state = self.full_dataset.iloc[self.period_start:self.period_start+self.period_length]
        
        self.state = scale_data(self.state,new_fit=True)


    def get_rel_change(self,prediction_variable='Close'):
        next_day = self.get_next_day(scale=False)
        last_day_of_state = scale_data(self.scaler,self.state,inverse_transform=True)[prediction_variable].iloc[-1]
        if last_day_of_state == 0:
            relative_change = next_day[prediction_variable].iloc[0]/self.eps 

        relative_change = (next_day[prediction_variable].iloc[0]-last_day_of_state)/last_day_of_state
        return relative_change

    def reset_state(self,):
        self.state = self.full_dataset.iloc[:self.period_length]
        self.state = scale_data(self.scaler,self.state,new_fit=True)

        self.period_start=0
        

    def get_reward(self,action):
        if action == 0: #do nothing
            return 0
        rel_change = self.get_rel_change()
        if action == 1: #buy
            return rel_change
        if action == 2: #sell
            return -rel_change
        
        
class Rl_env_multiple():
    def __init__(self,sec_codes,scaler,period_length=10,columns=['Close'],scale_everything = False):
        self.full_dataset=read_and_prepare_data(sec_codes,columns)
        self.index=self.full_dataset.index
        print(f'loaded {self.full_dataset.shape[0]} rows from codes {sec_codes}')
        self.full_dataset.plot()

        self.scaler = scaler
        self.period_length = period_length
        self.period_start = 0

        if scale_everything:
            self.unscaled_dataset = self.full_dataset.copy()
            self.full_dataset = scale_data(self.scaler,self.full_dataset,new_fit = True)
            self.full_dataset.plot()

        self.reset_state()
        self.current_distribution = np.ones(len(sec_codes))/len(sec_codes)

        # self.current_value = self.calc_value(self.current_distribution,self.state.iloc[-1])

    def calc_value(self,distribution,state):
        value = distribution * state
        return value.sum()

    def step_period(self,):
        self.period_start+=1
        self.state = self.full_dataset.iloc[self.period_start:self.period_start+self.period_length]

    def reset_state(self,):
        self.state = self.full_dataset.iloc[:self.period_length]
        self.period_start=0
        
    # def get_rel_change(self):
    #     next_day = self.get_next_day(scale=False)
    #     last_day_of_state = scale_data(self.state,inverse_transform=True).iloc[-1]
    #     if last_day_of_state == 0:
    #         relative_change = next_day.iloc[0]/self.eps 

    #     relative_change = (next_day.iloc[0]-last_day_of_state)/last_day_of_state
    #     return relative_change

    def get_next_day(self,scale=False,inv_scale=False):
        row = self.full_dataset.iloc[self.period_start+self.period_length]
        if scale:
            return scale_data(self.scaler,row,new_fit=False)
        if inv_scale:
            return scale_data(self.scaler,row,inverse_transform=True)
        else:
            return row

    def get_reward_and_set_distribution(self,distribution):
        old_value = self.calc_value(distribution,self.state.iloc[-1])
        self.current_distribution = distribution
        new_value = self.calc_value(self.current_distribution,self.get_next_day())
        return new_value-old_value
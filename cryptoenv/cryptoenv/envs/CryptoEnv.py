import gym
from gym import spaces
import glob
import numpy as np
import pandas as pd
import random
import os 

from IPython.display import clear_output, display

# TODO

# prices nicht skewen? 
# non negative balance wie an agent feedbacken? 
# wie sicherstellen, dass nur innerhalb des balance gekauft wird? 
# starten immer mit random timestep oder von anfang an 
# turbulence mit angepasster reward function 
# cut action von [-1,1]? necessary? 
# flatten observation? sonst läuft stable baseline nicht 

PROCESSEDDATA = os.path.dirname(os.path.abspath(__file__)) + "/../../../processeddata"

INITIAL_BALANCE = 2000
# for normalization
MAX_BALANCE = 1000000 
MAX_PRICE = 10000 
MAX_INDEX_VAL = 1000 
MAX_HOLDING_VAL = 10000  
MAX_BUYING_AMOUNT = 10000 
TRANSACTION_PERCENTAGE_COST = 0.001

class CryptoEnv(gym.Env):
    
    def __init__(self):
    
        super(CryptoEnv, self).__init__()
        
        self.crypto_data = [] # ['unixtime', 'interpolateddata', 'open', 'close', 'low', 'high', 'macd', 'rsi', 'cci', 'adx']
        self.crypto_data_names = [] # ['ADA', 'ETH', 'XRP', 'XMR', 'LTC']
        self._read_processed_data()
        
        # TODO 
        self.action_space = spaces.Box(low=-1,high=+1,shape=(len(self.crypto_data_names),), dtype=np.float32)
        # for each crypto (normalized from -1 to +1): negative: sell, 0: hold, positive: buy 
        #self.action_space = spaces.Box(low=0.1,high=+0.9998,shape=(len(self.crypto_data_names),), dtype=np.float32)
        
        # TODO 
        obs_lows = [0] + [0] * len(self.crypto_data_names) * 2 + [-1] * len(self.crypto_data_names) + [0] * len(self.crypto_data_names) * 2 + [-1] * len(self.crypto_data_names)
        obs_highs =[1] + [1] * len(self.crypto_data_names) * 6 
        #self.observation_space = spaces.Box(low = np.array(obs_lows), high = np.array(obs_highs), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=((len(self.crypto_data_names) * 6 + 1),), dtype=np.float32)
       
        # all normalized from -1 to 1                                     
        # balance (pure money) (>= 0, single number)
        # amount owned of each crypto (>= 0, vector)
        # close price of each crypto (>= 0, vector)
        # MACD of each crypto (neg or pos, vector)
        # RSI of each crypto (>= 0, vector)
        # CCI of each crypto (>= 0, vector)
        # ADX of each crypto (neg or pos, vector)
        
        
        # init used variables, are reset in reset()
        self.current_step = 0
        self.current_balance = 0
        self.current_holdings = np.ones((len(self.crypto_data_names),)) # np.ones((len(self.crypto_data_names),)) * -1
        self.current_observation = np.array([np.array(0.6),np.array([[0.4]*(len(self.crypto_data_names))]*6)], dtype=object) #self.current_observation = -1 
        
        print("Initialized CryptoEnv.")
        

    def _read_processed_data(self):
        for crypto_file in glob.glob(PROCESSEDDATA + "/*.csv"):
            self.crypto_data.append(pd.read_csv(os.path.join(crypto_file), index_col=0))
            name = os.path.basename(crypto_file)
            name = name.split("BTC")[0]
            self.crypto_data_names.append(name)
        
        # drop some unimportant stuff 
        for idx, _ in enumerate(self.crypto_data):
            self.crypto_data[idx].drop(columns=["timestamp","openbtc","lowbtc","highbtc","closebtc"], inplace = True)
            self.crypto_data[idx].drop(columns=["high","low","open","interpolateddata"], inplace = True)
    
    
    def _perform_action(self,action):
        
        
        # Implementation:
        # Sell individual holdings where possible 
        # Only buy all that are planned to be bought or none  
        # Buy only if the max holding is not overflowed 
        
        
        # De-normalize 
        action = action * MAX_BUYING_AMOUNT 
        
        # total buy price
        total_buy_price = 0
        
        # exlude these idx from buying, because they would overflow max holding 
        exclude_idx_buying = []
        
        # performs sells if possible immediately, count all buys together to evaluate if possible 
        for idx, amount in enumerate(action):
            
            
            
            # want to sell
            if (amount < 0): 
                abs_amnt = np.abs(amount)
                if (self.current_holdings[idx] < abs_amnt):  # less crypto available in holdings than wanted to sell 
                    pass # do not sell and keep balance
                else: # crypto available
                    self.current_holdings[idx] -= abs_amnt # sell, remove from holdings
                    self.current_balance += (abs_amnt * self.current_observation[1][1][idx] * MAX_PRICE) * (1 - TRANSACTION_PERCENTAGE_COST / 100) # add money to balance, deduct transaction cost; get price from observation
            
            # in case of hold do nothing 
            
            # want to buy 
            elif (amount > 0):
                
                # make sure that max holding would not be overflowed 
                if (self.current_holdings[idx] + amount) <= MAX_HOLDING_VAL: 
                    total_buy_price += amount * self.current_observation[1][1][idx] * MAX_PRICE # add price of wanted cryptos to total buy price 
                else:
                    exclude_idx_buying.append(idx)
        
        # when buys are possible, perform
        if total_buy_price <= self.current_balance:
            
            for idx, amount in enumerate(action):
                
                if (amount > 0 and (idx not in exclude_idx_buying)): # make sure buying does not exceed max holding
                    self.current_holdings[idx] += amount # buy, add to holdings
                    self.current_balance -= (amount * self.current_observation[1][1][idx] * MAX_PRICE) * (1 + TRANSACTION_PERCENTAGE_COST / 100) # remove money from balance, deduct transaction cost; get price from observation
                    if self.current_balance < 0:
                        self.current_balance = 0 # in case transaction cost was too expensive, clip to 0 


        
            
    def _calc_portfolio_value(self):

        return self.current_balance + np.squeeze(np.dot(self.current_holdings,np.squeeze(self.current_observation[1][1] * MAX_PRICE)))
        

    def step(self, action):
        
        # Log
        if (self.current_step % 100 == 0 and False):
            clear_output(wait=True)
            print("Timestep: " + str(self.current_step) + " Datetime: " + str(pd.to_datetime(self.crypto_data[0].loc[self.current_step]['unixtime'],unit='s')))
            print("Current balance: " + str(self.current_balance))
            for idx, holding in enumerate(self.current_holdings):
                print(self.crypto_data_names[idx] + " holding: " + str(holding) + "\t" + self.crypto_data_names[idx] + " price: " + str(self.current_observation[1][1][idx] * MAX_PRICE) + " normalized price: " + str(self.current_observation[1][1][idx]))
        
        
        
        portfolio_val_before = self._calc_portfolio_value()
        
        # Execute one time step within the environment
        self._perform_action(action)
        
        portfolio_val_after = self._calc_portfolio_value()
        
        reward = portfolio_val_after - portfolio_val_before # transaction costs are already included in the portfolio 

        self.current_step += 1

        done = False 
            
        # Episode done if balance below 0 
        if self.current_balance < 0:
            print("Balance below 0")
            done = True 
        
        # Epsiode done if reached end of data
        if self.current_step >= len(self.crypto_data[0]['close'] - 1): 
            print("Reached end of data")
            done = True
        
        
        obs = self._observe_step()
        

        return obs, reward, done, {}
        
    
    
    # all uncommented here commented -> remove comment, error occurs
    # when perform action is uncommented -> almost none so it's probably the combination of these two
    def _observe_step(self):
         
        crypto_info_frames = []
        
        # check if end of data is reached 
        reached_end = False
        if self.current_step >= len(self.crypto_data[0]['close'] - 1): 
            reached_end = True 
            self.current_step -= 1 # stay at last time step, env will be reset after this observation by gym (from step()) 
            print("lowered step by one")
        
        # extract relevant data for all cryptos 
        for idx, _ in enumerate(self.crypto_data):            
            crypto_info_frames.append(self.crypto_data[idx].iloc[self.current_step])
        crypto_info_df = pd.concat(crypto_info_frames, axis=1)
        crypto_info_df = crypto_info_df.T
        crypto_info_np = crypto_info_df.to_numpy()
        crypto_info_np = np.array(crypto_info_np[:,1:])
        # column 0: close, 1: macd, 2: rsi, 3: cci, 4: adx 
        # 
        # debugging
        #print("crypto_info_df:")
        #print(crypto_info_df)
        #print("crypto_info_np (removed unix time) (column 0: close, 1: macd, 2: rsi, 3: cci, 4: adx):")
        #print(crypto_info_np)
        
        
        # normalize 
        crypto_info_np[:,0] = crypto_info_np[:,0] / MAX_PRICE
        crypto_info_np[:,1] = crypto_info_np[:,1] / MAX_INDEX_VAL
        crypto_info_np[:,2] = crypto_info_np[:,2] / MAX_INDEX_VAL
        crypto_info_np[:,3] = crypto_info_np[:,3] / MAX_INDEX_VAL
        crypto_info_np[:,4] = crypto_info_np[:,4] / MAX_INDEX_VAL
        # debugging
        #print("crypto_info_np (normalized):")
        #print(crypto_info_np)
        
        # concat balance and prices and indices 
        observation = np.vstack([self.current_holdings / MAX_HOLDING_VAL,crypto_info_np.T])
        observation = np.array([[self.current_balance / MAX_BALANCE], observation], dtype=object)
        # observation make up 
        # [current balance]
        # [ [current holdings 1, 2, 3, ...]
        #   [ close 1, 2, 3, ... ] 
        #   [ macd 1, 2, 3, ... ]  
        #   [ rsi 1, 2, 3, ... ] 
        #   [ cci 1, 2, 3, ... ] 
        #   [ adx 1, 2, 3, ... ] 
        # ]
        #
        # debugging
        #print("Current balance: " + str(self.current_balance) + " Current balance normalized: " + str(self.current_balance / MAX_BALANCE))
        #print("Current holdings: \n" + str(self.current_holdings) + "\n Current holdings normalized: \n" + str(self.current_holdings / MAX_HOLDING_VAL))
        #print("Full observation shape: " + str(np.shape(observation)))
        #print("Observation index 0 shape: " + str(np.shape(observation[0])))
        #print("Observation index 0 content:")
        #print(observation[0])
        #print("Observation index 1 shape: " + str(np.shape(observation[1])))
        #print("Observation index 1 content: (row 0: current holding, 1: close, 2: macd, 3: rsi, 4: cci, 5: adx)")
        #print(observation[1])
        
        # go back to true timestep 
        if reached_end:
            self.current_step += 1
            print("increased step by one")
        
        self.current_observation = observation 
        
        # make observation into single vector for return 
        return np.insert(observation[1].flatten(), 0, observation[0], axis=0)
    
    
    
    
    def reset(self):
        
        # Reset to a random number from start until half of available data points 
        self.current_step = random.randint(
            0, len(self.crypto_data[0]['close']) // 2)
        
        self.current_balance = INITIAL_BALANCE
        self.current_holdings = np.zeros((len(self.crypto_data_names)),) # start with no cryptos 
        # current prices, macd, rsi, cci, adx are obtained from the data 
        
        # Observe with the newly set values 
        observation = self._observe_step()
        
        return observation 

    
    def render(self, mode='human', close=False):
               
        # Render the environment to the screen
        pass


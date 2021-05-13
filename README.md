# DRLCrypto

A deep reinforcement learning framework for trading of cryptocurrency, based on the paper [Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996) by Yang, Liu, Zhong & Walid



## Contributions
- Data retrieval
- Data preprocessing
- Custom environment 
- Custom PPO implementation

## Setting up
To run the agent, a conda environment with the dependencies listed inside the `environment.yml` should be created.

Additionally, `pip install -e cryptoenv` should be run from the main directory to register the custom environment with Gym. 

## Starting the agent
From the main directory, `python run_ppo_agent.py` will start the agent using our custom PPO implementation.

## File overview 
- The `DataCollection.ipynb` notebook can be used to acquire crypto chart data. It is based on the Binance API, so in order to use it, an account with Binance needs to be created and the API key and secret need to be added to the notebook. Data can be acquired in varying time bins (e.g. one hour or one day.) We decided to acquire data for Cardano, Ethereum, Litecoin, Monero and Ripple. The acquired data is saved in the `rawdata` folder.  
- The `DataPreprocessing.ipynb` notebook can be used to clean up the acquired data (e.g. to fill holes in the time series with means), to calculate the USD prices of the coins from the Bitcoin prices and to calculate indices that are used during training of the agent. Specifically these are the moving average convergence divergence (MACD), relative strength index (RSI), commodity channel index (CCI) and the average directional index (ADX). The processed data is saved in the `processeddata` folder. 
- The custom gym environment is located inside the `cryptoenv` folder, more specifically in `cryptoenv/cryptoenv/envs/CryptoEnv.py`. The folder and file structure surrounding it is needed for the environment to be accessible by Gym. Inside the `cryptoenv` folder, there is the `dev_environment.ipynb` notebook which holds the environment as well and is meant for developing and quick adaption purposes. It also holds a [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) PPO training routine for quick-checking. 
- The custom PPO implementation is located in the `run_ppo_agent.py` script. 
- Information about a short training run of the custom PPO agent can be found in the `progress_ppo` and `modelProgress` folders. 



## Acknowledgments
The data acquisition and index calculations are built upon existing sources, the respective notebooks contain the credits. 

Our implementation of PPO makes use of the [ReAllY](https://github.com/geronimocharlie/ReAllY) framework.


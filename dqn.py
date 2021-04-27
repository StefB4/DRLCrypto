# do not worry if your loss doesn't decrease significantly! As the agent should see new samples in each iteration the loss should stay about stable
# make use of a replay buffer -> think about the relation between the number of steps you sample and the number of steps you sample from the buffer in each iteration (tip -> the latter should be significantly bigger than the first) (you might also want to first fill your buffer completely before starting to sample to avoid overfitting)
# you may want to anneal your value for epsilon for the epsilon-greedy sampling -> it makes sense to enforce more exploration in the beginning and more exploitation later on in the training progress
# in the vanilla version it is normal that deep q is somewhat unstable
# you probably want to make use of the already implemented methods in the really framework of the agent -> agent.q_val(state, action), agent.max_q(state)
# again do not forget to pass on the optimized weights to the manager to collect new samples (manager.set_agent(new_weights))
# the actual tensorflow q network doesnt need to be that big! 2-3 layers with 16-32 units dependent on your activation function (tanh, leaky_relu...) should be enough
# in the readout layer of your q_values you should however not use any activation function and at best no bias, the number of neurons in the readout layer of course should correspond to the size of your action space.


# Get model

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets,
)

class VanillaDQN():
    
    def __init__(self, output_units=2):
           
        self.model = keras.Sequential(name="dqn")
        self.model.add(keras.Input(shape=(4,)))
        self.model.add(layers.Dense(32, activation="relu", name="layer1"))
        self.model.add(layers.Dense(16, activation="relu", name="layer2"))
        self.model.add(layers.Dense(2, activation = None, bias = None, name="output"))
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def __call__(self, x_in):

        output = {}
        action = self.model(x_in)
        output["action"] = action

        return output
    
    def get_weights(self):
        pass
    
    def set_weights(self):
        pass


if __name__ == "__main__":

    kwargs = {
        "model": VanillaDQN,
        "environment": "CartPole-v0",
        "num_parallel": 5,
        "total_steps": 100,
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": 1,
    }

    ray.init(log_to_driver=False)

    manager = SampleManager(**kwargs)
    # where to save your results to: create this directory in advance!
    saving_path = os.getcwd() + "/dqn_progress"

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    # print("test before training: ")
    # manager.test(test_steps, do_print=True)

    # get initial agent
    agent = manager.get_agent()
    # print(agent.model.trainable_variables)

    for e in range(epochs):

        # training core

        # experience replay
        print("collecting experience..")
        #Powerful multithread call
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample data to optimize on from buffer
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        # create and batch tf datasets
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

        print("optimizing...")

        for key, value in enumerate(data_dict):
            print("key: ", key)
            print("value: ", value)
            pass
        # TODO: iterate through your datasets

        # TODO: optimize agent
        
# NAIVE DQN PSEUDOCODE
# initialize:  Qnet(s,a)
# for each episode:

#     s0 = initial state
#     for each time step t:

#      sample action at
#      obtain reward rtand next state st+1
#      compute qtarget=yt=rt+γmaxat+1Q(st+1,at+1)
#      train Qnet(s,a) via backpropagation minimizing (yt−Qnet(st,at))²
#                st=st+1
# Notes:
# - sampling the action seems trivial, but one must make sure to ensure exploration here, e.g. via a  ϵ−greedy or Thompson-Sampling strategy

#  - the target yt we use to train Qnet(s,a) is actually dependent on Qnet(s,a) itself. This approach is called bootstrapping. Make sure to fix yt and detach it from any gradient computation before you use it for optimization!


        new_weights = agent.model.get_weights()

        # set new weights
        manager.set_agent(new_weights)
        # get new weights
        agent = manager.get_agent()
        # update aggregator
        time_steps = manager.test(test_steps)
        # manager.update_aggregator(loss=dummy_losses, time_steps=time_steps)
        # # print progress
        # print(
        #     f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in dummy_losses])}   avg env steps ::: {np.mean(time_steps)}"
        # )

        # yeu can also alter your managers parameters
        manager.set_epsilon(epsilon=0.99)

        if e % saving_after == 0:
            # you can save models
            manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
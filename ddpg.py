import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import gym
import ray
from really import SampleManager  
from really.utils import (
    dict_to_dict_of_datasets,
)  
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
 

class Actor(tf.keras.Model):
    def __init__(self):
      super(Actor, self).__init__()
      self.layer_1 = layers.Dense(32, activation="relu", input_shape = (8,))
      self.layer_2 = layers.Dense(16, activation="relu")
      self.mu_layer = layers.Dense(2, activation= None, use_bias=False)
      self.sigma_layer = layers.Dense(2, activation= None, use_bias=False)
        
    
    def __call__(self, state_in):

        output = {}
        layer_stack = self.layer_2(self.layer_1(state_in))
        mu = self.mu_layer(layer_stack)
        sigma = self.sigma_layer(layer_stack)

        
        return mu, sigma
    
    
    def get_weights(self):
        return super().get_weights()
    
    def set_weights(self, weights):
        return super().set_weights(weights)
    
    
##################################################################################################
class Critic(tf.keras.Model):
    def __init__(self, input_units=4, output_units=2):
        super(Critic, self).__init__()

        # State as input
        state_input = layers.Input(shape=(8,))
        state_out = layers.Dense(32, activation="relu")
        state_out2 = layers.Dense(16, activation="relu")

        # Action as input
        action_input = layers.Input(shape=(1,))
        action_out = layers.Dense(32, activation="relu")


        out = layers.Dense(64, activation="relu")
        out2 = layers.Dense(32, activation="relu")
        output = layers.Dense(1, activation = None)

        # self.model = tf.keras.Model([state_input, action_input], output)

    def __call__(self, state_input, action_input):
        state = self.state_out2(self.state_out(self.state_input(state_input)))
        action = self.action_out(self.action_input(action_input))
        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state, action])
        
    
        # output = {}
        return self.output(self.out2(self.out(concat)))
        
    
    def get_weights(self):
        
        return super().get_weights()
    
    
    def set_weights(self, weights):
        
        return super().set_weights(weights)
    
        

#################################################################################################################
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
            
############################################################################################################
class DDPG(tf.keras.Model):
    def __init__(self, actor_input_units=8, actor_output_units=2, critic_input_units=10, critic_output_units=2):
        super(DDPG, self).__init__()



        self.actor = Actor()
        self.critic = Critic()
        self.target_actor = Actor()
        self.target_critic = Critic()
        
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        self.gamma = 0.9

    def __call__(self, state_in):

        output = {}
        
        
        mu, sigma = self.actor(state_in)
        print("TYPE MU SIGMA ", mu, sigma)
        # q_value = self.critic(state_in, action)
        output["mu"] = mu
        output["sigma"] = sigma
        
        return output
    
    def get_weights(self):
        return self.actor.get_weights()

    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value =  self.critic([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss,  self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad,  self.critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value =  self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
        
        return actor_loss, critic_loss

    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
##################################################################################################################


##################################################################################################

if __name__ == "__main__":

    # constants 
    TAU = 0.99
    EPSILONSTART = 1
    EPSILONDECAYRATE = 0.005
    EPSILONMIN = 0.01
    DELAYED_UPDATE = 20

    buffer_size = 5000
    test_steps = 200
    epochs = 20
    sample_size = 100 #1000
    optim_batch_size = 8
    saving_after = 5

    kwargs = {
        "model": DDPG,
        "environment": "LunarLanderContinuous-v2",
        "num_parallel": 5,
        "total_steps": 200, # 100
        "action_sampling_type": "continuous_normal_diagonal",
        "num_episodes": 20, # per runner
    }

    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)
    saving_path = os.getcwd() + "/progress_ddpg"


    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    # initialize buffer
    manager.initilize_buffer(buffer_size, optim_keys)

    # initilize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "time_steps"]
    )

    # initial testing:
    #print("test before training: ")
    #manager.test(test_steps, do_print=True, render=False)

    # get initial agent
    agent = manager.get_agent()
    #print(agent.model.trainable_variables)

    # keep track of buffer size and epsilon decay 
    number_of_elems_in_buffer = 0
    epsilon_decay_started_at_e = 0 

    for e in range(epochs):

        print("\nStarting epoch " + str(e+1))
        sampleNumber = 0

        # experience replay
        while (number_of_elems_in_buffer < buffer_size):
            print("Collecting experience..")
            data = manager.get_data()
            manager.store_in_buffer(data)
            number_of_elems_in_buffer += len(data['reward'])
            print("Now " + str(number_of_elems_in_buffer) + " elements in buffer. Waiting until buffer is filled before sampling for optimization.")

            
            
        print("Buffer full (" + str(buffer_size) +  " elements), saw additional " + str(number_of_elems_in_buffer - buffer_size) + " elements. Sampling now for optimization.")

        if epsilon_decay_started_at_e == 0:
            epsilon_decay_started_at_e = e

        # sample data to optimize on from buffer
        while sampleNumber < (buffer_size // sample_size):
            sample_dict = manager.sample(sample_size)
            # create and batch tf datasets
            #data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

            print("Optimizing on sample")

            state_batch = tf.convert_to_tensor(sample_dict["state"] )
            action_batch =tf.convert_to_tensor( sample_dict["action"])
            reward_batch= tf.convert_to_tensor(sample_dict["reward"])
            next_state_batch = tf.convert_to_tensor(sample_dict["state_new"])
            # sample_dict["not_done"]
            
            
            # Get last lost 
            actor_loss, critic_loss = agent.model.update(state_batch, action_batch, reward_batch, next_state_batch)

            if sampleNumber % DELAYED_UPDATE == 0:
                        agent.model.update_target(agent.model.target_actor.variables, agent.model.actor.variables, TAU)
                        agent.model.update_target(agent.model.target_critic.variables, agent.model.critic.variables, TAU)
            sampleNumber += 1
                

            

        # Get optimized weights and update agent
        new_weights = agent.model.get_weights()
        manager.set_agent(new_weights)
        agent = manager.get_agent()

        # update aggregator
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=actor_loss, time_steps=time_steps)
        print(f"Epoch : {e}  Loss : {np.mean([np.mean(l) for l in actor_loss])}   Avg env steps ::: {np.mean(time_steps)}")


        if e % saving_after == 0:
            manager.save_model(saving_path, e)


    # test after training
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
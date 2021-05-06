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
from tensorflow.keras import Model

from cryptoenv import CryptoEnv

class A2C(Model):
    def __init__(self, layers, action_dim):
        super(A2C, self).__init__()
        self.mu_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Policy_mu_{i}'
                ) for i, num_units in enumerate(layers)]

        self.readout_mu = tf.keras.layers.Dense(units=action_dim,
                                                activation=None,
                                                name='Policy_mu_readout'
                                                )

        self.sigma_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Policy_sigma_{i}'
                ) for i, num_units in enumerate(layers)]
                
        self.readout_sigma = tf.keras.layers.Dense(units=action_dim,
                                                   activation=None,
                                                   name='Policy_sigma_readout'
                                                   )

        self.value_layer = [
            tf.keras.layers.Dense(
                units=num_units,
                activation='relu',
                name=f'Value_layer_{i}'

                ) for i, num_units in enumerate(layers)]
                
        self.readout_value = tf.keras.layers.Dense(units=1,
                                                   activation=None,
                                                   name='Value_readout'
                                                   )

    @tf.function
    def call(self, input_state):
        output = {}
        mu_pred = input_state
        sigma_pred = input_state
        value_pred = input_state
        for layer in self.mu_layer:
            mu_pred = layer(mu_pred)
        for layer in self.sigma_layer:
            sigma_pred = layer(sigma_pred)
        for layer in self.value_layer:
            value_pred = layer(value_pred)

        # Actor
        output["mu"] = tf.squeeze(self.readout_mu(mu_pred))
        output["sigma"] = tf.squeeze(tf.abs(self.readout_sigma(sigma_pred)))
        # Critic
        output["value_estimate"] = self.readout_value(value_pred)
        return output
    
  

if __name__ == "__main__":
    
    #env = gym.make("LunarLanderContinuous-v2")
    env = gym.make("cryptoenv-v0")
    
    print("Environment built.")
    
    model_kwargs = {"layers": [32,32,32], "action_dim": env.action_space.shape[0]}



    ENV_ID              = "cryptoenv-v0"
    TARGET_REWARD       = 2500
    CLIP_PARAM          = 0.2
    
    CRITIC_DISCOUNT     = 0.5
    ENTROPY_BETA        = 0.005
    GAMMA               = 0.99
    GAE_LAMBDA          = 0.95
    LEARNING_RATE       = 1e-4
    
    NUM_PARALLEL        = 4
    EPOCHS              = 30
    ELEMENTS_IN_BATCH   = 10
    NUM_BATCHES         = 20
    NUM_RUNNER_STEPS    = 30
    NUM_RUNNER_EPISODES = 1
    CONTINUE_TRAINING   = False



    kwargs = {
        "model": A2C,
        "model_kwargs": model_kwargs,
        "is_tf": True,
        "environment": ENV_ID,
        "num_parallel": NUM_PARALLEL,
        "total_steps": 200, # 100 Only for ER, no necessity in PPO
        "action_sampling_type": "continuous_normal_diagonal",
        # "num_episodes": NUM_RUNNER_EPISODES,
        "num_steps": NUM_RUNNER_STEPS,
        "returns": ['value_estimate', 'log_prob', 'monte_carlo'],
    }

    
    
    
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)
    

    mse_loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)


    saving_path = os.getcwd() + "/progress_ppo"
    # initialize progress aggregator
    manager.initialize_aggregator(
        path=saving_path, saving_after=5, aggregator_keys=["loss", "reward", "time"]
    )

    print("Aggregator built.")
    # get initial agent
    if(CONTINUE_TRAINING):
        agent = manager.load_model("./modelProgress")
    else:
        print("Building new model.")
        agent = manager.get_agent()


    rewards = []

    print("Starting training.")
    print()
    
    for e in range(EPOCHS):

        print("\nStarting Epoch " + str(e+1) +":")
        print()


        print("Sampling: INPROCESS")
        sample_dict = manager.sample(
            sample_size = NUM_BATCHES*ELEMENTS_IN_BATCH,
            from_buffer = False
            )
        print("Sampling: DONE")
        sample_dict['value_estimate'].append(agent.v(np.expand_dims(sample_dict['state_new'][-1],0)))
        
        print("GAE: INPROCESS")
        sample_dict['advantage'] = []
        gae = 0
        # Loop backwards through rewards
        for i in reversed(range(len(sample_dict['reward']))):
            delta = sample_dict['reward'][i] + GAMMA * sample_dict['value_estimate'][i+1].numpy() * sample_dict['not_done'][i] - sample_dict['value_estimate'][i].numpy()
            gae = delta + GAMMA * GAE_LAMBDA * sample_dict['not_done'][i] * gae
            # Insert advantage in front to get correct order
            sample_dict['advantage'].insert(0, gae)
        # Center advantage around zero
        sample_dict['advantage'] -= np.mean(sample_dict['advantage'])

        print("GAE: DONE")
        print()
        
        
        
        # Remove keys that are no longer used
        sample_dict.pop('value_estimate')
        sample_dict.pop('state_new')
        sample_dict.pop('reward')
        sample_dict.pop('not_done')            
        # create and batch tf datasets
        samples = dict_to_dict_of_datasets(sample_dict, batch_size=ELEMENTS_IN_BATCH)
        


        print("Start optimizing on sample:")
        
        actor_losses = []
        critic_losses = []
        losses = []
        elementNumber = 0
        for state_batch, action_batch, advantage_batch, returns_batch, log_prob_batch in zip(samples['state'], samples['action'], samples['advantage'], samples['monte_carlo'], samples['log_prob']):
            # print("Processing batch:", elementNumber)
            with tf.GradientTape() as tape:                
            
                old_log_prob = tf.cast(log_prob_batch, dtype=tf.float32)
                new_log_prob, entropy = agent.flowing_log_prob(state_batch, action_batch, True)
                advantage_batch = tf.cast(advantage_batch, dtype=tf.float32)
                ratio = tf.exp(new_log_prob - old_log_prob)
                # print("Calculated ratio: ", ratio)
                ppo1 = ratio * tf.expand_dims(advantage_batch,1)
                # print("Advantage: ", tf.expand_dims(advantage_batch,1))
                # print("PPO1", ppo1)
                ppo2 = tf.clip_by_value(ratio, 1-CLIP_PARAM, 1+CLIP_PARAM) * tf.expand_dims(advantage_batch,1)
                # print("PPO2", ppo2)
                
                actor_loss = -tf.reduce_mean(tf.minimum(ppo1,ppo2),0)

                value_target = tf.cast(returns_batch, dtype=tf.float32)
                value_pred = agent.v(state_batch)
                critic_loss = mse_loss(value_target,value_pred)
    
                #Maybe invert Vorzeichen
                total_loss = actor_loss + CRITIC_DISCOUNT * critic_loss  - ENTROPY_BETA * entropy
            
            
                gradients = tape.gradient(total_loss, agent.model.trainable_variables)

            optimizer.apply_gradients(zip(gradients, agent.model.trainable_variables))
            
            
            actor_losses.append(actor_loss)                
            critic_losses.append(critic_loss)                
            losses.append(total_loss) 
            
            manager.set_agent(agent.get_weights())



        # if (e+1) % 5 == 0:
        print('TESTING...')
        steps, current_rewards = manager.test(
            max_steps=100,
            test_episodes=10,
            render=False,
            evaluation_measure="time_and_reward",
            )
        print("DONE")

    
        # Update aggregator
        manager.update_aggregator(loss=losses, reward=current_rewards, time=steps)
        
        # Collect all rewards
        rewards.extend(current_rewards)
        # Average reward over last 100 episodes
        avg_reward = sum(rewards[-100:])/min(len(rewards),100)


        print("OUTCOME:")
        # Print progress   
           
        print(
            f"      Epoch ::: {e+1}   Loss ::: {np.mean(losses)} avg_current_reward ::: {np.mean(current_rewards)} avg_reward ::: {avg_reward} avg_timesteps ::: {np.mean(steps)}"
            )
               
              
            

        if False: # avg_reward > env.spec.reward_threshold:
            print(f'\n\nEnvironment solved after {e+1} episodes!')
            # Save model
            manager.save_model(saving_path, e, model_name='LunarLanderContinuous')
            break
        print("-------------------------------------------------------------------------------------------------------------------------------------------------------")
    print()
    print("Finished with optimization.")
    print("Testing optimized agent...")
    manager.test(
        max_steps=100,
        test_episodes=10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )
    
    manager.save_model("./modelProgress", EPOCHS, model_name="ppo")
    print("END")
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.utils import ReplayBuffer, OUNoise

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.layers import BatchNormalization, Dropout


# Create DDPG Actor    
class Actor:
    
    def __init__(self, state_size, action_size, action_low, action_high, load=False):
        """Initialize Parameters and build model"""
        self.state_size = state_size  # integer - dimension of each state
        self.action_size = action_size  # integer - dimension of each action
        self.action_low = action_low  # array - min value of action dimension
        self.action_high = action_high  # array - max value of action dimension
        self.action_range = self.action_high - self.action_low
        self.load_model = load
        
        self.build_model()

    def func(self, x):
        import tensorflow as tf
        # shape = # scale factor shape
        # scale_factor = np.random.normal(shape) * std + avg
        return (x * self.action_range) + self.action_low
    
    def build_model(self):
        """Create actor network that maps states to actions"""
        
        # Define input states
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # Create hidden layers
        # net = layers.Dense(units=32, activation='relu')(states)
        # net = layers.Dense(units=64, activation='relu')(net)
        # net = layers.Dense(units=32, activation='relu')(net)
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None)

        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        # net = BatchNormalization()(net)
        net = Dropout(0.2)(net)
        net = layers.Dense(units=64, activation='relu')(net)
        # net = BatchNormalization()(net)
        net = Dropout(0.2)(net)
        net = layers.Dense(units=32, activation='relu')(net)
        # net = BatchNormalization()(net)
        # net = Dropout(0.2)(net)


        # final_layer_initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)
        
        # raw_actions = layers.Dense(units=self.action_size, activation='tanh', name='raw_actions', kernel_initializer=final_layer_initializer)(net)


        # Output layer with sigmoid
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
        
        # Scale output for each action to appropriate ranges
        actions = layers.Lambda(self.func, name='actions')(raw_actions)

        # Note that the raw actions produced by the output layer are in a [-1.0, 1.0] range 
        # (using a 'tanh' activation function). So, we add another layer that scales each 
        # output to the desired range for each action dimension, where the middle value of 
        # the action range corresponds to the value in the middle of the tanh function, which 
        # is 0. This produces a deterministic action for any given state vector.
        # middle_value_of_action_range = self.action_low +self.action_range/2
        # actions = layers.Lambda(lambda x: (x * self.action_range) + middle_value_of_action_range,
        #     name='actions')(raw_actions)
        
        # Create model
        if self.load_model:
            # load json and create model
            # with open('/media/lyn/Samsung_T5/actor_model.h5', 'r') as json_file:
            #     loaded_model_json = json_file.read()
            #     print(loaded_model_json)
            # json_file.close()
            # self.model = model_from_json(loaded_model_json)
            self.model = load_model('/media/lyn/Samsung_T5/actor_model.h5', custom_objects={"func": self.func})
            model_json = self.model.to_json()
            with open("/media/lyn/Samsung_T5/actor_model_json.json", "w") as json_file:
                json_file.write(model_json)
        else:        
            self.model = models.Model(inputs=states, outputs=actions)
            # Loss function using Q-val gradients
            action_grads = layers.Input(shape=(self.action_size,))
            loss = K.mean(-action_grads * actions)
            
            # Optimizer and Training Function
            optimizer = optimizers.Adam()
            updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
            self.train_fn = K.function(inputs=[self.model.input, action_grads, K.learning_phase()], outputs=[], updates=updates_op)
            
        
        
        

# Create DDPG Critic
class Critic:
    
    def __init__(self, state_size, action_size, load=False):
        """Initialize parameters and model"""
        
        self.state_size = state_size  # integer - dim of states
        self.action_size = action_size  # integer - dim of action
        self.load_model = load

        self.build_model()
        
    def build_model(self):
        """Critic network for mapping state-action pairs to Q-vals"""
        
        # Define inputs
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')


        # Kernel initializer with fan-in mode and scale of 1.0
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='uniform', seed=None)
        # Kernel L2 loss regularizer with penalization param of 0.01
        kernel_regularizer = tf.keras.regularizers.l2(0.01)


        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        # net_states = BatchNormalization()(net_states)
        net_states = Dropout(0.2)(net_states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        # net_states = BatchNormalization()(net_states)
        net_states = Dropout(0.2)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        # net_actions = BatchNormalization()(net_actions)
        net_actions = Dropout(0.2)(net_actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        # net_actions = BatchNormalization()(net_actions)
        # net_actions = Dropout(0.2)(net_actions)

        # Combine state and action values
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # net = layers.Dense(units=64, activation='relu', kernel_initializer=kernel_initializer)(net)
            
        # Kernel initializer for final output layer: initialize final layer weights from 
        # a uniform distribution of [-0.003, 0.003]
        # final_layer_initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003, seed=None)

        # Q_values = layers.Dense(units=1, activation=None, name='q_values', kernel_initializer=final_layer_initializer, kernel_regularizer=kernel_regularizer)(net)


        # DEPRECATED: Hidden layers for states
        # net_states = layers.Dense(units=32, activation='relu')(states)
        # net_states = layers.Dense(units=64, activation='relu')(net_states)
            
        # Hidden layers for actions
        # net_actions = layers.Dense(units=32, activation='relu')(actions)
        # net_actions = layers.Dense(units=64, activation='relu')(net_actions)
            
            
        # Output layer for Q-values OLD
        Q_vals = layers.Dense(units=1, name='q_vals')(net)
            
        # Create model
        if self.load_model:
            # load json and create model
            # with open('/media/lyn/Samsung_T5/critic_model_json.json', 'r') as json_file:
            #     loaded_model_json = json_file.read()
            #     print(loaded_model_json)
            #     json_file.close()
            # self.model = model_from_json(loaded_model_json)
            self.model = load_model('/media/lyn/Samsung_T5/critic_model.h5')
            model_json = self.model.to_json()
            with open("/media/lyn/Samsung_T5/critic_model_json.json", "w") as json_file:
                json_file.write(model_json)
            print('model loaded')
        else:        
            self.model = models.Model(inputs=[states, actions], outputs=Q_vals)
            # Define Optimizer and compile
            optimizer = optimizers.Adam()
            self.model.compile(optimizer=optimizer, loss='mse')
                
        # Compute Q' wrt actions
        action_grads = K.gradients(Q_vals, actions)
        print(actions.shape)
        for item in action_grads:
            print(type(item))
            
        # Create function to get action grads
        self.get_action_gradients = K.function(
            inputs=[self.model.input, K.learning_phase()],
            outputs=action_grads)
        
        
          
        
        

class DDPG(BaseAgent):
    
    def __init__(self, task):
        
        self.task = task
        self.load = False
        # Constrain State and Action matrices
        self.state_size = 3
        self.action_size = 3
        # For debugging:
        print("Constrained State {} and Action {}; Original State {} and Action {}".format(self.state_size, self.action_size,  
            self.task.observation_space.shape, self.task.action_space.shape))
        
        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.3
        
        # Save episode statistics for analysis
        self.stats_filename = os.path.join(util.get_param('out'), "stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = ['episode', 'total_reward']
        self.episode_num = 1
        print("Save Stats {} to {}".format(self.stats_columns, self.stats_filename))
        
        # Actor Model
        self.action_low = self.task.action_space.low[0:3]
        self.action_high = self.task.action_space.high[0:3]
        
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, True)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        
        # Critic Model
        self.critic_local = Critic(self.state_size, self.action_size, True)
        self.critic_target = Critic(self.state_size, self.action_size)
        
        if self.load:
            # Initialize model parameters with local parameters
            self.critic_local.model.load_weights('/media/lyn/Samsung_T5/critic_weightsV4.h5')
            self.actor_local.model.load_weights('/media/lyn/Samsung_T5/actor_weightsV4.h5')
            print('model loaded')

        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # if not self.load:
        #     # Initialize model parameters with local parameters
        #     self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        #     self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        #     print('model built')
        # else:
        #     self.critic_target.model.load_weights('/media/lyn/Samsung_T5/critic_weights.h5')
        #     self.actor_target.model.load_weights('/media/lyn/Samsung_T5/actor_weights.h5')
        #     print('model loaded')
        
        # Process noise
        self.noise = OUNoise(self.action_size)
        
        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)
        
        # ALGORITHM PARAMETERS
        self.gamma = 0.99  # discount
        self.tau = 0.001  # soft update of targets
        
        # Episode vars
        self.reset_episode_vars()
    
    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        # self.task.reset()
        # self.noise.reset() 
        
    def step(self, state, reward, done):
        
        # Reduce state vector
        state = self.preprocess(state)
        
        # Choose an action
        action = self.act(state)
        

        # Save experience and reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward
            self.count += 1
        
        if done:
            # Learn from memory
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)
            # Write episode stats and reset
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1
            print(self.total_reward)


            # if np.sum(action < 0.01):
            #     print(action)
            #     self.noise.reset()

            # Save weights
            if self.episode_num == 500 or self.episode_num == 1000 or self.episode_num == 3000:
                self.save_model()
            self.reset_episode_vars()

        
        
        
        self.last_state = state
        self.last_action = action
        
        # Returns completed action vector
        return self.postprocess(action)
    
    def act(self, states):
        """Returns actions for a given state for current policy"""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        new_action = actions + self.noise.sample()

        return new_action
    
    def learn(self, experiences):
        """Update policy and value parameters given experiences"""
        # Convert experiences to separate arrays for each element
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        
        # Get predicted next actions and Q-vals from target model
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])
        
        # Compute Q targets for current state and train critic
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
        
        # Train actor model
        print(states.shape, actions.shape)
        test = self.critic_local.get_action_gradients([states, actions])
        action_gradients = np.reshape(test, (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])
        
        # Soft-update target model
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)
        
    def soft_update(self, local_model, target_model):
        """Soft update model params"""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())
        
        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
        
    def write_stats(self, stats):
        """Write an episode of stats to a CSV file"""
        df_stats = pd.DataFrame([stats], columns=self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a', index=False, header=not os.path.isfile(self.stats_filename))
    
    def preprocess(self, state, state_size=3):
        """Return state vector of just linear position and velocity"""
#         state = np.concatenate(state[0:3], state[7:10])
        state = state[0:3]
        return state
    
    def postprocess(self, action, action_size=3):
        """Return action vector of linear forces by default"""
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[0:action_size] = action
        return complete_action
    
    def save_model(self):
        self.actor_target.model.save_weights('/media/lyn/Samsung_T5/actor_weightsV4.h5')
        self.critic_target.model.save_weights('/media/lyn/Samsung_T5/critic_weightsV4.h5')

        self.critic_target.model.save('/media/lyn/Samsung_T5/critic_model.h5') 
        self.actor_target.model.save('/media/lyn/Samsung_T5/actor_model.h5') 

        print('Weights Saved. EP:', self.episode_num)

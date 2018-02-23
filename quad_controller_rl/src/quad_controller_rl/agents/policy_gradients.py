import os

import random
import numpy as np
import pandas as pd

from collections import namedtuple, deque

from keras import layers, models, optimizers, initializers
from keras import backend as K

from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl import util


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

# Create ReplayBuffer
class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""
    
    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        
        self.size = size
        self.memory = deque(maxlen=self.size)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        
        return len(self.memory)

# Create OUNoise
class OUNoise:
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, size, mu=None, theta=0.15, sigma=0.3):
        """Initialize parameters and noise process."""
        
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        
        return self.state


# Create DDPG Actor    
class Actor:
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize Parameters and build model
		
	Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size  # integer - dimension of each state
        self.action_size = action_size  # integer - dimension of each action
        self.action_low = action_low  # array - min value of action dimension
        self.action_high = action_high  # array - max value of action dimension
        self.action_range = self.action_high - self.action_low
        
        self.build_model()
    
    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        
		# Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)
        
        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid', name='raw_actions')(net)
        
        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low, name='actions')(raw_actions)
        
        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)
        
        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)
        
        # Define Optimizer and Training Function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()], outputs=[], updates=updates_op)


# Create DDPG Critic
class Critic:
    """Critic (Value) Model."""
    
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        
        self.state_size = state_size  # integer - dim of states
        self.action_size = action_size  # integer - dim of action
        
        self.build_model()
        
    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
		# Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
            
        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
            
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
            
        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
            
        # Add final output layer to prduce action values (Q values)
        Q_vals = layers.Dense(units=1, name='q_vals')(net)
            
        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_vals)
          
        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
            
        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_vals, actions)
            
        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)

# Create DDPG Agent
class DDPG(BaseAgent):
    """Reinforcement Learning Agent that learns using DDPG."""

    def __init__(self, task):

        self.task = task
        self.state_size = 3  # position only
        self.state_range = self.task.observation_space.high - self.task.observation_space.low
        self.action_size = 3  # force only
        self.action_range = (self.task.action_space.high - self.task.action_space.low)[0:self.action_size]

        # Actor (Policy) model
        self.action_low = self.task.action_space.low[0:self.action_size]
        self.action_high = self.task.action_space.high[0:self.action_size]
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.0  # Discount factor
        self.tau = 0.001  # for soft update of target parameters

        self.reset_episode_vars()

        # Save episodes stats
        self.stats_filename = os.path.join(util.get_param('out'), "stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = ['episode', 'total_reward']
        self.episode_num = 1

        print("Saving stats {} to {}".format(self.stats_columns, self.stats_filename))


    def preprocess_state(self, state):
        """Reduce state vector to relevant dimensions."""
        return state[0:3]


    def postprocess_action(self, action):
        """Return complete action vector."""
        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[0:3] = action
        return complete_action


    def reset_episode_vars(self):
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0


    def step(self, state, reward, done):
        # Reduce the state vector
        state = self.preprocess_state(state)

        # Choose an action
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)

        # Learn if enough sample are available in the memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

        self.last_state = np.copy(state)
        self.last_action = np.copy(action)
        self.total_reward += reward

        if done:
            print("Score: {}".format(self.total_reward))

            # Write episode stats
            self.write_stats([self.episode_num, self.total_reward])
            self.episode_num += 1

            # Reset episode vars
            self.reset_episode_vars()

        
        return self.postprocess_action(action)


    def act(self, states):
        """Return actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        return actions + self.noise.sample() # add some noise for exploration


    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
		
        # Convert experience tuples to separate arrays for each element
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next_state actions and Q values from target model
        action_next = self.actor_target.model.predict_on_batch(next_states)
        q_target_next = self.critic_target.model.predict_on_batch([next_states, action_next])

        # Compute Q targets for current states and train local critic model
        q_targets = rewards + self.gamma * q_target_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=q_targets)

        # Train local actor model
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])

        # Soft update target model
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)



    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""

        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)



    def write_stats(self, stats):
        """Write single episode stats to CSV file"""

        df_stats = pd.DataFrame([stats], columns=self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a', index=False, header=not os.path.isfile(self.stats_filename))


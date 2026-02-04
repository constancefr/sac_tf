'''
Components:
- replay buffer
- state value function network V
    - params psi
    - modeled as expressive NNs
- soft Q-function network Q
    - params theta
    - modeled as expressive NNs
- tractable policy network pi
    - params phi
    - modeled as Gaussian with mean and covariance given by NNs

- batch norm?

SAC Hyperparams (from paper abstract):
--Shared--
    optimiser: Adam
    learning rate: 3e-4
    discount factor: 0.99
    replay buffer size: e6
    num hidden layers per network: 2
    num hidden units per layer: 256
    num samples per minibatch: 256
    nonlinearity: ReLU
--SAC--
    target smoothing coeff: 0.005
    target update interval: 1
    gradient steps: 1
--SAC hard target update)--
    target smoothing coeff: 1
    target update interval: 1000
    gradient steps: 4 (except humanoids)
    gradient steps: 1 (humanoids)

'''
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
TODO:
- accelerate operations?
'''

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        '''
        We want our buffer to store states, new states, actions, rewards, and terminal states.
        '''
        self.mem_size = max_size
        self.mem_counter = 0 # keep track of most recently saved memory
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, new_state, action, reward, done):
        index = self.mem_counter % self.mem_size # override oldest mems (wraps around)
        self.state_memory[index] = state
        self.action_memory[index] = action # note that actions themselves are arrays
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = 1 - int(done) # want reward*0 when episode terminates
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal


class ActorNetwork(nn.Module):
    def __init__(self, ): # pass remainder args
        '''
        Define network layers
        '''

    def forward(self, state):
        '''
        Specify how data passes through the network
        '''
    
class CriticNetwork(nn.Module):
    def __init__(self, ): # pass remainder args
        '''
        Define network layers
        '''

    def forward(self, state):
        '''
        Specify how data passes through the network
        '''

class Agent(object):
    def __init__(self, ): # pass remainder args
        '''
        Init replay buffer, networks, optimisers
        Copy weights from main to target networks
        '''
    
    def update_network_params(self, ):
        
    def choose_action(self, ):
        
    def remember(self, ):

    def learn(self, ):
        
    def save_models(self, ):

    def load_models(self, ):

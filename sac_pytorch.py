'''
Components:
- replay buffer
- 3 networks: actor, critic, value
    - state value function network V
        - params psi
        - modeled as expressive NNs
    - soft Q-function network Q
        - params theta
        - modeled as expressive NNs
    - tractable policy network pi
        - params phi
        - modeled as Gaussian with mean and covariance given by NNs
            -> we don't directly output actions, but rather a mean and stdv for a distribution
            that we'll sample to get our action.

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
import torch.optim as optim
from torch.distributions.normal import Normal

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

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_counter % self.mem_size # override oldest mems (wraps around)
        self.state_memory[index] = state
        self.action_memory[index] = action # note that actions themselves are arrays
        self.reward_memory[index] = reward
        self.new_state_memory[index] = new_state
        self.terminal_memory[index] = float(done)
        # self.terminal_memory[index] = 1 - int(done) # want reward*0 when episode terminates?
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones


class ActorNetwork(nn.Module):
    '''
    Sample a probability distribution with mean and covariance generated from the network
    (rather than a simple feed-forward operation).
    '''
    def __init__(self, alpha, # learning rate
                 input_dims, max_action,
                 fc1_dims=256, fc2_dims=256, n_actions=2,
                 name="actor", chkpt_dir="tmp/sac"
                 ):
        '''
        Define network layers - each network has 2 hidden layers
        '''
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_sac")
        self.max_action = torch.as_tensor(max_action, dtype=torch.float32)
        # self.max_action = max_action # for env action bounds
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions) # mean of our policy's distribution 
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions) # stdv

        self.optimiser = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

        # paper does not mention any specific weight initialisation, so leave default (uniform)

    def forward(self, state):
        '''
        Specify how data passes through the network
        '''
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu=self.mu(prob)
        log_std = self.sigma(prob) # TODO: rename sigma to log_std?

        log_std = torch.clamp(log_std, min=-20, max=2)

        return mu, log_std
    
    def sample_normal(self, state, reparametrise=True):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if not torch.isfinite(mu).all():
            print("NaNs in mu", mu)
        if not torch.isfinite(std).all():
            print("NaNs in std", std)

        probabilities = Normal(mu, std)

        if reparametrise:
            actions = probabilities.rsample() # adding some noise
        else:
            actions = probabilities.sample()

        # Equation 21 (appendix C - enforcing action bounds)
        u = actions
        a = torch.tanh(u) # in [-1, 1]

        log_probs = probabilities.log_prob(u) # for loss function (originally sampled actions)
        log_probs -= torch.log(1-a.pow(2) + self.reparam_noise) # avoid log(0)
        log_probs = log_probs.sum(1, keepdim=True) # to convert from n_actions dim to scalar
        
        # Rescale action to env bounds
        action = a * self.max_action
        # action = a * torch.tensor(self.max_action).to(self.device)

        return action, log_probs

    def save_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    
class CriticNetwork(nn.Module):
    def __init__(self, beta,  # learning_rate
                 n_actions, input_dims, fc1_dims=256, fc2_dims=256,
                 name="critic", chkpt_dir='tmp/sac'):
        '''
        Define network fully connected layers.
        '''
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(
            in_features=self.input_dims[0]+n_actions, # critic evaluates state-action pair!
            out_features=self.fc1_dims
        )
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1) # scalar

        self.optimiser = optim.Adam(self.parameters(), lr=beta) # optimise over our params
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        '''
        Specify how data passes through the network.
        '''
        action_value = self.fc1(torch.cat([state, action], dim=1)) # concat state-action
        action_value = F.relu(action_value) # activation
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, 
                 input_dims, fc1_dims=256, fc2_dims=256,
                 name="value", chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        # no need for n_actions - value function is indep of actions
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimiser = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class Agent():
    def __init__(self, alpha=0.0003, beta=0.0003, tau=0.005, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau # for a soft copy
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        self.actor = ActorNetwork(alpha, input_dims, max_action=env.action_space.high,
                                  n_actions=n_actions, name="actor")
        self.critic_1 = CriticNetwork(beta, n_actions, input_dims, name="critic_1")
        self.critic_2 = CriticNetwork(beta, n_actions, input_dims, name="critic_2")
        self.value = ValueNetwork(beta, input_dims, name="value")
        self.target_value = ValueNetwork(beta, input_dims, name="target_value")

        self.scale = reward_scale
        self.update_network_parameters(tau=1) # set params of targ value network exactly = to start

    def choose_action(self, observation):
        state = torch.Tensor([observation]).to(self.actor.device) # convet obs to torch tensor
        actions, _ = self.actor.sample_normal(state, reparametrise=False)
        return actions.cpu().detach().numpy()[0] # take from cuda / graph to cpu
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def update_network_parameters(self, tau=None):
        '''
        TODO: make more elegant
        '''
        if tau is None:
            tau = self.tau

        with torch.no_grad():
            for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
                # Polyak update: target <- tau * online + (1 - tau) * target
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

    def save_models(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        print("Saving models...")
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print("Loading models...")
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return
        
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        
        reward = torch.tensor(reward, dtype=torch.float).to(self.actor.device)
        done = torch.tensor(done, dtype=torch.long).to(self.actor.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.actor.device)
        state = torch.tensor(state, dtype=torch.float).to(self.actor.device)
        action = torch.tensor(action, dtype=torch.float).to(self.actor.device)

        value = self.value(state).view(-1) # collapse along batch dim (avoid tensor w/in tensor)
        new_value = self.target_value(new_state).view(-1)
        new_value[done] = 0.0

        # Sample values of actions according to new policy (for actor & value networks)
        # TODO: put this chunk below in a separate function!!
        actions, log_probs = self.actor.sample_normal(state, reparametrise=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
            # take min of the two critics - improves stability of learning (overestimation bias)
        critic_value = critic_value.view(-1)
        # ----------

        # Calc value network loss & backpropagate
        self.value.optimiser.zero_grad()
        value_target = (critic_value - log_probs).detach()
        value_loss = 0.5 * F.mse_loss(value, value_target) # equation 5 (squared residual error)
        value_loss.backward() 
        # value_loss.backward(retain_graph=True) 
            # need to keep track of graph for losses of value and actor networks
        self.value.optimiser.step()

        # Actor network loss
        # TODO: put this chunk below in a separate function!!
        actions, log_probs = self.actor.sample_normal(state, reparametrise=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        # ----------

        actor_loss = log_probs - critic_value # ??
        actor_loss = torch.mean(actor_loss)
        self.actor.optimiser.zero_grad()
        actor_loss.backward()
        # actor_loss.backward(retain_graph=True)
        self.actor.optimiser.step()

        # Critic loss
        self.critic_1.optimiser.zero_grad() # reset gradients
        self.critic_2.optimiser.zero_grad()
        q_hat = self.scale*reward + self.gamma*new_value # equation 8
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
            # action from replay buffer (old policy) 
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat) # equation 7
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimiser.step()
        self.critic_2.optimiser.step()

        self.update_network_parameters()
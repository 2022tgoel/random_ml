import gym
#from DeepQLearner import Agent
#from utils import plotLearning
import numpy as np
from gym import wrappers
import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.to(self.device)
    
    def forward(self, observation):
        state = T.FloatTensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Agent(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=1000000, eps_min=0.01, eps_dec=0.996):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_size = max_mem_size
        self.mem_cntr = 0
        self.Q_eval = DeepQNetwork(lr, input_dims, fc1_dims=256, fc2_dims=256, n_actions=self.n_actions)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float64) 
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float64)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float64)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int64)
    
    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal
        self.new_state_memory[index] = state_
        self.mem_cntr += 1
    
    def choose_action(self, observation):
        rand = np.random.random()
        if (rand < self.epsilon):
            action = np.random.choice(self.action_space)
        else:
            actions = self.Q_eval(observation)
            action = T.argmax(actions).item()
        return action

    def learn(self):
        self.Q_eval.train()
        if (self.mem_cntr > self.batch_size):
            self.Q_eval.optimizer.zero_grad()
            max_mem = min(self.mem_cntr, self.mem_size)
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            state_batch = self.state_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            action_batch = T.LongTensor(self.action_memory[batch]).to(self.Q_eval.device)
            reward_batch = T.Tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            terminal_batch = self.terminal_memory[batch]
            q_eval = T.gather(self.Q_eval(state_batch), dim=1, index=action_batch.unsqueeze(1)).squeeze()
            q_next = self.Q_eval(new_state_batch)
            q_next[terminal_batch] = 0.0
            q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]
            loss = self.Q_eval.loss(q_target, q_eval.to(self.Q_eval.device))
            loss.backward()
            self.Q_eval.optimizer.step()
            self.epsilon = max(self.epsilon*self.eps_dec, self.eps_min)



env = gym.make('LunarLander-v2')
brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, input_dims=[8], lr=0.005)
scores = []
eps_history = []
score = 0
for i in range(500):
    if (i % 10 == 0 and i > 0):
        avg_score = np.mean(scores[i-10:])
        print('episode ', i, 'average score %.3f' % avg_score, 'epsilon %.3f' % brain.epsilon)
    score = 0
    eps_history.append(brain.epsilon)
    observation = env.reset()
    done = False
    while not done:
        action = brain.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        brain.store_transition(observation, action, reward, observation_, done)
        brain.learn()
        observation = observation_
    scores.append(score)
from collections import deque, namedtuple
import random
import numpy
import torch
import torch.nn as nn

class DQN(nn.Module):
  '''
  DQN

  References:
  https://github.com/mnovitzky/moos-ivp-pLearn/blob/master/pLearn/learning_code/DeepLearn.py
  https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
  https://github.com/seungjaeryanlee/implementations-nfq/blob/master/nfq/agents.py
  '''
  def __init__(self, state_size, n_actions):
    super().__init__()

    self.n_actions = n_actions

    self.model = nn.Sequential(
      nn.Linear(state_size, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, n_actions)
    )
  
  def forward(self, x):
    return self.model(x)

Transition = namedtuple(
  'Transition',
  ('state', 'action', 'next_state', 'reward', 'done')
)

class ReplayMemory:
  def __init__(self, capacity):
    self.memory = deque([],maxlen=capacity)

  def push(self, *args):
    """Save a transition"""
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

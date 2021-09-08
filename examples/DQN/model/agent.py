import numpy as np
import torch

class Agent:
  def __init__(self, model):
    self.model = model

  def select_action(self, state, epsilon=0.0):
    if np.random.random() < epsilon:
      return np.random.randint(0, self.model.n_actions)
    
    with torch.no_grad():
      return self.model(state).max(1)[1].item()
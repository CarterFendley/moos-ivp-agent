#!/usr/bin/env python3
import os
import time

from numpy import dtype
import wandb
import argparse
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.optim as optim

from mivp_agent.manager import MissionManager
from mivp_agent.util.display import ModelConsole
from mivp_agent.util.math import dist
from mivp_agent.aquaticus.const import FIELD_BLUE_FLAG

from wandb_key import WANDB_KEY

from DQN import DQN, ReplayMemory, Transition
from agent import Agent
from util.state import dist

from config import EPISODES, BATCH_SIZE, TARGET_UPDATE, MEMORY_SIZE
from config import GAMMA, LR
from config import EPS_START, EPS_DECAY_AMT
from config import REWARD_STEP, REWARD_SUCCESS, REWARD_FAILURE, REWARD_ACTION_CHANGE
from config import ACTIONS
from config import SAVE_DIR

# The expected agents and their enemies
VEHICLE_PAIRING = {
  'agent_11': 'drone_21',
  'agent_12': 'drone_22',
  'agent_13': 'drone_23',
  'agent_14': 'drone_24',
  'agent_15': 'drone_25'
}
EXPECTED_VEHICLES = []
for agent in VEHICLE_PAIRING:
  EXPECTED_VEHICLES.append(agent)
  EXPECTED_VEHICLES.append(VEHICLE_PAIRING[agent])

# TODO: Are these reasonable values? 
COLLECT_FOR = 25 # Units: Episodes
TRAIN_FOR = 50 # Units: btaches 

class AgentData:
  '''
  Used to encapsulate the data needed to run each indivdual
  agent, track state / episode transitions, and output useful
  information and statistics.
  '''
  def __init__(self, vname, enemy):
    self.vname = vname
    self.enemy = enemy
    
    # For running of simulation
    self.agent_episode_count = 0
    self.last_episode_num = None # Episode transitions
    self.last_state = None       # State transitions
    self.current_action = None 
    self.last_action = None

    # For debugging / output
    self.min_dist = None
    self.episode_reward = 0
    self.last_MOOS_time = None
    self.MOOS_deltas = []
  
  def new_episode(self, last_num):
    self.last_episode_num = last_num
    self.agent_episode_count += 1

    self.last_state = None
    self.current_action = None
    self.last_action = None

    self.min_dist = None
    self.episode_reward = 0
    self.last_MOOS_time = None
    self.MOOS_deltas.clear()
    self.times_switched = 0 


# References: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
def train(args, config):
  # ---------------------------------------
  # Part 1: Setup model
  policy_net = DQN(7, 2)
  target_net = DQN(7, 2)
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval() # Not 100% on purpose, but references DQN has BatchNorm, which I am not using https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
  
  model = Agent(policy_net)
  optimizer = optim.Adam(policy_net.parameters(), lr=config['lr'])
  memory = ReplayMemory(MEMORY_SIZE)
  
  vehicles = {}
  with MissionManager() as mgr:
    # ---------------------------------------
    # Part 2: Setup simulation & global data structures
    print('Waiting for sim vehicle connection...')
    mgr.wait_for(EXPECTED_VEHICLES)

    # Construct agent data object with enemy specified
    agents = {}
    for a in VEHICLE_PAIRING:
      agents[a] = AgentData(a, VEHICLE_PAIRING[a])

    # Initalize global vars
    total_batches = 0
    episode_count = 0
    epsilon = config['eps_start']
    progress_bar = tqdm(total=config['episodes'], desc='Training')

    while episode_count < config['episodes']:
      # ---------------------------------------
      # Part 3: Setup simulation & data structures needed to collect some data

      # While all vehicle's pEpisodeManager are not PAUSED
      tqdm.write('\nWaiting for pEpisodeManager to enter PAUSED state...')
      while not all(mgr.episode_state(vname) == 'PAUSED' for vname in agents):
        msg = mgr.get_message()
        if mgr.episode_state(msg.vname) != 'PAUSED':
          # Instruct them to stop and pause
          msg.stop()
        else:
          # O/w just keep up to date on their state
          msg.request_new()

      # Initalize the epidode numbers from pEpisodeManager
      for vname, num in mgr.episode_nums().items():
        if vname in agents:
          agents[vname].last_episode_num = num
      
      # ---------------------------------------
      # Part 4: Actually collect data

      # Setup statistics for collection
      collected = 0
      success_count = 0
      transition_count = 0
      action_changes = 0
      rewards = []
      durations = [] # TODO: Just .clear() these
      deltas = []
      action_count = [0]*args['action_space_size']

      while collected < COLLECT_FOR:
        msg = mgr.get_message()

        # Start un started vehicles
        if msg.episode_state == 'PAUSED':
          msg.start()
          continue

        # If not an agent, just ask for new message and no response
        if msg.vname not in agents:
          msg.request_new()
          continue

        agent = agents[msg.vname]

        # Update debugging MOOS time
        if agent.last_MOOS_time is not None:
          deltas.append(msg.state['MOOS_TIME']-agent.last_MOOS_time)
        agent.last_MOOS_time = msg.state['MOOS_TIME']
      
        # Construct model readable tensor
        dist_between = abs(dist(
          (msg.state['NAV_X'], msg.state['NAV_Y']),
          (msg.state['NODE_REPORTS'][agent.enemy]['NAV_X'], msg.state['NODE_REPORTS'][agent.enemy]['NAV_Y'])
        ))
        model_state = torch.Tensor([(
          msg.state['NAV_X'],
          msg.state['NAV_Y'],
          msg.state['NAV_HEADING'],
          msg.state['NODE_REPORTS'][agent.enemy]['NAV_X'],
          msg.state['NODE_REPORTS'][agent.enemy]['NAV_Y'],
          msg.state['NODE_REPORTS'][agent.enemy]['NAV_HEADING'],
          dist_between # Don't want the net to have to learn a dist function
        ),])

        # Catch episode endings
        success = None
        if msg.episode_report is None:
          assert agent.agent_episode_count == 0 # Sanity check
        elif msg.episode_report['NUM'] != agent.last_episode_num:
          # Get reward
          reward = config['reward_failure']
          if msg.episode_report['SUCCESS']:
            reward = config['reward_success']
            success_count += 1
          agent.episode_reward += reward

          memory.push(
            agent.last_state,
            agent.current_action,
            None,
            reward,
          )

          # Log info
          rewards.append(agent.episode_reward)
          durations.append(msg.episode_report['DURATION'])
          episode_report = {
            'reward': agent.episode_reward,
            'duration': round(msg.episode_report['DURATION'], 2),
            'success': msg.episode_report['SUCCESS'],
          }
          tqdm.write(f'[{msg.vname}] ', end='')
          tqdm.write(', '.join([f'{k}: {episode_report[k]}' for k in episode_report]))

          # Update epsilon
          if epsilon > 0:
            epsilon -= config['eps_decay']

          # Reset agent data
          agent.new_episode(msg.episode_report['NUM'])

          # Update counters
          transition_count +=1
          episode_count += 1
          collected += 1
          progress_bar.update(1)

        # Add a previous transition to memory
        if agent.last_state != None:
          reward = config['reward_step']
          if agent.last_action != agent.current_action:
            reward = config['reward_action_change']
            action_changes += 1

          memory.push(
            agent.last_state,
            agent.current_action,
            model_state,
            reward
          )

          agent.episode_reward += reward
          transition_count += 1
        agent.last_state = model_state

        # Get a new action
        agent.last_action = agent.current_action
        agent.current_action = model.select_action(model_state, epsilon=epsilon)

        # Preform the action
        msg.act(ACTIONS[agent.current_action])
        action_count[agent.current_action] += 1
      
      # ---------------------------------------
      # Part 4: Construct report for previous collection session
      report = {
        'episode_count': episode_count,
        'epsilon': round(epsilon, 3),
        'prob_success': round(success_count/COLLECT_FOR, 2),
        'prob_action': round(action_changes / transition_count, 2),
        'avg_reward': round(sum(rewards)/len(rewards)),
        'avg_duration': round(sum(durations)/len(durations), 2),
        'avg_deltas': round(sum(deltas)/len(deltas), 2),
        'prob_loiter1': round(action_count[0]/sum(action_count)),
        'prob_loiter2': round(action_count[1]/sum(action_count)),
      }

      # Log the report
      if not args.no_wandb:

        wandb.log(report)
      tqdm.write(', '.join([f'{k}: {report[k]}' for k in report]))


      # ---------------------------------------
      # Part 4: Fit the model
      # Reference: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training-loop
      batches = 0
      fit_bar = tqdm(total=TRAIN_FOR, desc='Fitting')
      tqdm.write(f'Batch size: {BATCH_SIZE}, Batches: {TRAIN_FOR}, Will Sample: {BATCH_SIZE*TRAIN_FOR}, Memory size: {len(memory)}')
      while batches < TRAIN_FOR:
        if len(memory) < BATCH_SIZE:
          break

        # Get a batch
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions)) # See Transition definition & https://stackoverflow.com/a/19343/3343043

        # Get V(s_{t+1}) expected value for next states
        non_terminal_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)

        non_terminal_next_states = torch.cat([s for s in batch.next_state if s is not None])

        next_state_values = torch.zeros(BATCH_SIZE) # Default will be zero for terminal states
        next_state_values[non_terminal_mask] = target_net(non_terminal_next_states).max(1)[0].detach() # Detach grad

        # Get Q value to fit
        rewards = torch.cat([torch.Tensor([r]) for r in batch.reward])
        Q_target = (GAMMA * next_state_values) + rewards

        # Get Q prediction
        states = torch.cat(batch.state)
        # NOTE: Double [[a]] to match dimensionality requirement of gather()
        actions = torch.cat([torch.tensor([[a]], dtype=torch.long) for a in batch.action])

        Q_prediction = policy_net(states).gather(1, actions) # Get only actions we are interested in

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_prediction, Q_target.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
        optimizer.step()

        batches += 1
        fit_bar.update(1)
      fit_bar.close()
    
    # Update target after training for x amount
    target_net.load_state_dict(policy_net.state_dict())


if __name__ == '__main__':
  save_dir = os.path.join(SAVE_DIR, str(round(time.time())))

  parser = argparse.ArgumentParser()
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--save_dir', default=save_dir)
  parser.add_argument('--no_wandb', action='store_true')
  
  args = parser.parse_args()

  # Error if wandb not set --no_wandb is not
  if not args.no_wandb and WANDB_KEY == "Your API key here":
    raise RuntimeError('WandB key not set and no --no_wandb flag. See documentation')

  # Construct config
  config = {
    'lr': LR,
    'gamma': GAMMA,
    'episodes': EPISODES,
    'eps_start': EPS_START,
    'eps_decay': EPS_DECAY_AMT,
    'actions': ACTIONS,
    'action_space_size': len(ACTIONS),
    'reward_success': REWARD_SUCCESS,
    'reward_failure': REWARD_FAILURE,
    'reward_step': REWARD_STEP,
    'reward_action_change': REWARD_ACTION_CHANGE,
  }

  if args.no_wandb:
    train(args, config)
  else:
    wandb.login(key=WANDB_KEY)
    with wandb.init(project='mivp_agent_dqn', config=config):
      config = wandb.config
      args.save_dir = os.path.join(SAVE_DIR, f'{str(round(time.time()))}_{wandb.run.name}')
      train(args, config)

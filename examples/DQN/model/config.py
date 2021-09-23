import os

LR = 0.0001
EPISODES = 10_000
HALF_BATCH_SIZE = 100
BATCH_SIZE=HALF_BATCH_SIZE*2
GAMMA = 0.7
EPS_START = 0.9
EPS_DECAY_END = EPISODES // 2 # Half way through
EPS_DECAY_AMT = EPS_START / EPS_DECAY_END
TARGET_UPDATE = 10
MEMORY_SIZE = 10_000

REWARD_STEP = -1
REWARD_ACTION_CHANGE = -5 # Change action
REWARD_SUCCESS = 1000
REWARD_FAILURE = -80

ACTION_LOITER1 = {
  'speed': 0.0,
  'course': 0.0,
  'posts': {
    'ACTION': 'LOITER1'
  }
}
ACTION_LOITER2 = {
  'speed': 0.0,
  'course': 0.0,
  'posts': {
    'ACTION': 'LOITER2'
  }
}
ACTIONS = [
  ACTION_LOITER1,
  ACTION_LOITER2
]


# Get this file's dirname for use later
DIRNAME = os.path.abspath(os.path.dirname(__file__))
SAVE_DIR = os.path.abspath(os.path.join(DIRNAME, '../../trained'))

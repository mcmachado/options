import sys
import time
import pygame
import numpy as np
import matplotlib.pylab as plt

from random import randrange
from ale_python_interface import ALEInterface

np.set_printoptions(threshold=np.nan)

def initializeALE(romFile):
  ale = ALEInterface()

  #max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
  ale.setInt("max_num_frames_per_episode", 18000)
  ale.setInt("random_seed", 123)
  ale.setFloat("repeat_action_probability", 0.0)
  ale.setInt("frame_skip", 5)

  random_seed = ale.getInt("random_seed")

  # Set USE_SDL to true to display the screen. ALE must be compilied
  # with SDL enabled for this to work. On OSX, pygame init is used to
  # proxy-call SDL_main.

  USE_SDL = True
  if USE_SDL:
    if sys.platform == 'darwin':
      pygame.init()
      ale.setBool('sound', False) # Sound doesn't work on OSX
    elif sys.platform.startswith('linux'):
      ale.setBool('sound', True)
    ale.setBool('display_screen', True)

  ale.loadROM(romFile)
  actionSet = ale.getMinimalActionSet()

  return ale, actionSet

# Read arguments
if(len(sys.argv) < 4):
  print("Usage ./ale_python_test1.py <ROM_FILE_NAME> <GAME_NAME> <TRAJECTORY_ID>")
  sys.exit()

# Initialize ALE
ale, actionSet = initializeALE(sys.argv[1])
gameName = sys.argv[2]
trajId = sys.argv[3]

startRandom = False
actionsTaken = []
total_reward = 0
while not ale.game_over():
  action = 0
  keys = pygame.key.get_pressed()

  if keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
    action = 14
  elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
    action = 15
  elif keys[pygame.K_DOWN]:
    action = 5
  elif keys[pygame.K_LEFT]:
    action = 4
  elif keys[pygame.K_RIGHT]:
   action = 3
  elif keys[pygame.K_UP]:
    action = 2
  elif keys[pygame.K_SPACE]:
    action = 1
  elif keys[pygame.K_RETURN]:
    startRandom = True

  # Apply an action and get the resulting reward
  if startRandom:
    action = actionSet[np.random.randint(actionSet.size)]
    reward = ale.act(action);
  else:
    reward = ale.act(action);
    actionsTaken.append(action)
  
  total_reward += reward

ale.reset_game()

np.save('humanStarts/' + gameName + '_' + trajId, np.asarray(actionsTaken))

print 'Final Reward:', total_reward
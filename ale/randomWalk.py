import sys
import time
import numpy as np
import matplotlib.pylab as plt

import RAMFeatures
import DataCollection

from ale_python_interface import ALEInterface

np.set_printoptions(threshold=np.nan)

def initializeALE(romFile):
  ale = ALEInterface()

  #max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
  ale.setInt("max_num_frames_per_episode", 18000)
  ale.setInt("random_seed", 123)
  ale.setFloat("repeat_action_probability", 0.0)

  random_seed = ale.getInt("random_seed")
  #Set record flags
  rec_dir = 'random'
  ale.setString(b'record_screen_dir', rec_dir + '/')
  ale.setString("record_sound_filename", rec_dir + "/sound.wav")
  #We set fragsize to 64 to ensure proper sound sync 
  ale.setInt("fragsize", 64)

  # Set USE_SDL to true to display the screen. ALE must be compilied
  # with SDL enabled for this to work. On OSX, pygame init is used to
  # proxy-call SDL_main.

  USE_SDL = False
  if USE_SDL:
    if sys.platform == 'darwin':
      import pygame
      pygame.init()
      ale.setBool('sound', False) # Sound doesn't work on OSX
    elif sys.platform.startswith('linux'):
      ale.setBool('sound', True)
    ale.setBool('display_screen', True)

  ale.loadROM(romFile)
  actionSet = ale.getMinimalActionSet()

  return ale, actionSet

# Read arguments
if(len(sys.argv) < 2):
  print("Usage ./ale_python_test1.py <ROM_FILE_NAME>")
  sys.exit()

# Initialize ALE
ale, actionSet = initializeALE(sys.argv[1])

while not ale.game_over():
# Random walk
  a = actionSet[np.random.randint(actionSet.size)]
  reward = ale.act(a);

ale.reset_game()

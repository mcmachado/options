import sys
import numpy as np
import ctypes
import matplotlib.pylab as plt

import RAMFeatures

from ale_python_interface import ALEInterface

np.set_printoptions(threshold=np.nan)

def initializeALE(romFile, rec_dir):
  ale = ALEInterface()

  max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
  ale.setInt("random_seed",123)
  ale.setFloat("repeat_action_probability", 0.0)
  ale.setInt("frame_skip", 5)
  #Set record flags
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
if(len(sys.argv) < 3):
  print("Usage ./ale_python_test1.py <ROM_FILE_NAME> <OPTION_TO_LOAD>")
  sys.exit()

# Initialize ALE
fileToLoad = sys.argv[2]
rec_dir = fileToLoad.split('/')[1].split('.')[0]
ale, actionSet = initializeALE(sys.argv[1], rec_dir)

policy = np.load(fileToLoad)

rewards = []

#print policy
#print len(policy)

'''
indices = np.where(policy == 1)
print indices, len(indices[0])

if len(indices[0]) > 40:
  '''
# We learn how to maximize such eigenpurposes
prevFeatures = np.zeros(RAMFeatures.NUM_BITS_RAM)
eigenpurposes = np.load('eigenpurposes.npy')
idx = int(fileToLoad.split('/')[1].split('.')[0])
eigenpurpose = eigenpurposes[idx]
frame = 0
while not ale.game_over() and frame < len(policy):
  #print frame

  nextAction = policy[frame]
  if nextAction == -1:
    extrinsicReward = ale.act(actionSet[0]); #No-Op
  else:
    extrinsicReward = ale.act(actionSet[nextAction]);

  currFeatures = RAMFeatures.getRAMVector(ale)
  featureVector = currFeatures - prevFeatures
  intrinsicReward = eigenpurpose.dot(featureVector)
  rewards.append(intrinsicReward)


  frame += 1
  prevFeatures = currFeatures

ale.reset_game()

plt.plot(xrange(0, 5*len(rewards), 5), rewards)
plt.xlabel('Time step')
plt.ylabel('Immediate Reward')
plt.title('Reward along the episode (Freeway)')
plt.savefig(rec_dir + '/' + rec_dir + '.pdf')



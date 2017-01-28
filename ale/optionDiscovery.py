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
  ale.setInt("frame_skip", 5)

  random_seed = ale.getInt("random_seed")
  print("random_seed: " + str(random_seed))

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
  print("Usage ./ale_python_test1.py <ROM_FILE_NAME> <MAX_NUM_FRAMES>")
  sys.exit()

# Initialize ALE
ale, actionSet = initializeALE(sys.argv[1])
maxNumFrames = int(sys.argv[2])


# Collect samples for SVD
T = DataCollection.buildTransitionMatrix(ale, actionSet, maxNumFrames)

# SVD over collected samples
U, s, V = np.linalg.svd(T)

# Each eigenpurpose is in a line now
eigenpurposes = V.transpose()
negEigenpurposes = -1.0 * V.transpose()
X = np.concatenate((eigenpurposes, negEigenpurposes))

# We save the eigenpurposes
np.save('eigenpurposes', eigenpurposes)
np.save('eigenvalues', s)

# We learn how to maximize such eigenpurposes
prevFeatures = RAMFeatures.getRAMVector(ale)

options = []
for i in xrange(len(s)):
  options.append([])

for i in xrange(len(s)):
  frame = 0
  eigenpurposeIdx = 1023 - i
  optionIdx = eigenpurposeIdx
  print 'Learning option ', eigenpurposeIdx

  while not ale.game_over():

    bestActionIdx = -1
    bestActionVal = 0

    # Try all actions 
    for actionIdx in xrange(len(actionSet)):
      ale.saveState()
      extrinsicReward = ale.act(actionSet[actionIdx]);
      currFeatures = RAMFeatures.getRAMVector(ale)
      featureVector = currFeatures - prevFeatures
      intrinsicReward = eigenpurposes[eigenpurposeIdx].dot(featureVector)
      if intrinsicReward > bestActionVal:
        bestActionVal = intrinsicReward
        bestActionIdx = actionIdx

      ale.loadState()

    if bestActionIdx == -1:
      extrinsicReward = ale.act(actionSet[0]);
    else:
      extrinsicReward = ale.act(actionSet[bestActionIdx]);

    prevFeatures = currFeatures
    options[optionIdx].append(bestActionIdx)
    frame += 1

  ale.reset_game()
  options[optionIdx] = np.asarray(options[optionIdx])
  np.save('eigenbehaviors/' + str(optionIdx), options[optionIdx])

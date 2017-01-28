import sys
import time
import numpy as np
import matplotlib.pylab as plt

import RAMFeatures
import DataCollectionHumanStarts

from ale_python_interface import ALEInterface

np.set_printoptions(threshold=np.nan)

def initializeALE(romFile):
  ale = ALEInterface()

  #max_frames_per_episode = ale.getInt("max_num_frames_per_episode");
  ale.setInt("max_num_frames_per_episode", 18000)
  ale.setInt("random_seed", 123)
  ale.setFloat("repeat_action_probability", 0.0)
  ale.setInt("frame_skip", 5)

  # Set USE_SDL to true to display the screen. ALE must be compilied
  # with SDL enabled for this to work. On OSX, pygame init is used to
  # proxy-call SDL_main.

  USE_SDL = True
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
if(len(sys.argv) < 4):
  print("Usage ./ale_python_test1.py <ROM_FILE_NAME> <MAX_NUM_FRAMES> <GAME>")
  sys.exit()

# Initialize ALE
ale, actionSet = initializeALE(sys.argv[1])
maxNumFrames = int(sys.argv[2])
gameName = sys.argv[3]

# Collect samples for SVD

setOfTransitions = set()
randomStart = np.load('humanStarts/' + gameName + '_0.npy')
setOfTransitions, T = DataCollectionHumanStarts.buildTransitionMatrix(ale, actionSet, maxNumFrames, setOfTransitions, randomStart)
print len(T)
for i in xrange(1, 6):
  randomStart = np.load('humanStarts/' + gameName + '_' + str(i) + '.npy')
  setOfTransitions, listTransitions = DataCollectionHumanStarts.buildTransitionMatrix(ale, actionSet, maxNumFrames, setOfTransitions, randomStart)
  T = np.concatenate((T, listTransitions))
  print len(T)
# SVD over collected samples (my parameters limit the number of rows to 21.6k)
U, s, V = np.linalg.svd(T)

# Each eigenpurpose is in a line now
eigenpurposes = V.transpose()
negEigenpurposes = -1.0 * V.transpose()
X = np.concatenate((eigenpurposes, negEigenpurposes))

# We save the eigenpurposes
np.save('eigenpurposes', eigenpurposes)
np.save('eigenvalues', s)


# We learn how to maximize such eigenpurposes for each random start
for trj in xrange(0, 6):
  randomStart = np.load('humanStarts/' + gameName + '_' + str(trj) + '.npy')

  options = []
  for i in xrange(len(s)):
    options.append([])

  for i in xrange(len(s)):
    frame = 0
    eigenpurposeIdx = 1023 - i
    optionIdx = eigenpurposeIdx
    print 'Learning option ', eigenpurposeIdx

    for a in xrange(len(randomStart)):
        extrinsicReward = ale.act(randomStart[a]);

    prevFeatures = RAMFeatures.getRAMVector(ale)

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
    np.save('eigenbehaviors_' + gameName + '/' + str(trj) + '_' + str(optionIdx), options[optionIdx])

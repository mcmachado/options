import sys
import time
import numpy as np
import matplotlib.pylab as plt

import RAMFeatures
import DataCollectionHumanStarts

from ale_python_interface import ALEInterface

np.set_printoptions(threshold=np.nan)

numOptions = 1024

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

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
  print("Usage ./computeActionGap.py <ROM_FILE_NAME> <GAME>")
  sys.exit()

# Initialize ALE
ale, actionSet = initializeALE(sys.argv[1])
gameName = sys.argv[2]

# Collect samples for SVD

# Each eigenpurpose is in a line. Importantly, the negation is not in here
eigenpurposes = np.load('fullHumanEigenpurposes_freeway.npy')

# We learn how to maximize such eigenpurposes for each random start
for trj in xrange(0, 1):
  randomStart = np.load('humanStarts/' + gameName + '_' + str(trj) + '.npy')

  options = []
  actionsVal = []

  for i in xrange(numOptions):
    options.append([])
    actionsVal.append([])
    for j in xrange(len(actionSet)):
      actionsVal[i].append([])

  for i in xrange(numOptions):
    frame = 0
    extrinsicReward = 0
    eigenpurposeIdx = 1023 - i
    optionIdx = eigenpurposeIdx
    print 'Learning option ', eigenpurposeIdx

    for a in xrange(len(randomStart)):
        extrinsicReward += ale.act(randomStart[a]);
        frame += 1

    prevFeatures = RAMFeatures.getRAMVector(ale)

    while not ale.game_over():
      bestActionIdx = -1
      bestActionVal = 0

      # Try all actions 
      for actionIdx in xrange(len(actionSet)):
        ale.saveState()
        extrinsicReward += ale.act(actionSet[actionIdx]);
        currFeatures = RAMFeatures.getRAMVector(ale)
        featureVector = currFeatures - prevFeatures
        intrinsicReward = eigenpurposes[eigenpurposeIdx].dot(featureVector)
        actionsVal[optionIdx][actionIdx].append(intrinsicReward)
        if intrinsicReward > bestActionVal:
          bestActionVal = intrinsicReward
          bestActionIdx = actionIdx

        ale.loadState()

      if bestActionIdx == -1:
        extrinsicReward += ale.act(actionSet[0]);
      else:
        extrinsicReward += ale.act(actionSet[bestActionIdx]);

      prevFeatures = currFeatures
      options[optionIdx].append(bestActionIdx)
      frame += 1

    ale.reset_game()
    options[optionIdx] = np.asarray(options[optionIdx])
    #np.save('eigenbehaviors_' + gameName + '/' + str(trj) + '_' + str(optionIdx), options[optionIdx])


    plt.plot(moving_average(actionsVal[optionIdx][0], n=400), label='no-op')
    plt.plot(moving_average(actionsVal[optionIdx][1], n=400), label='up')
    plt.plot(moving_average(actionsVal[optionIdx][2], n=400), label='down')
    plt.legend()
    plt.savefig('graphs/' + str(optionIdx) + '.pdf')
    plt.clf()

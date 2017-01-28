import numpy as np

import RAMFeatures

def buildTransitionMatrix(ale, actionSet, maxNumFrames):

	# Required initializations
	totalNumFrames = 0
	setTransitions = set()
	listTransitions = []

	# While I need to collect data:
	while totalNumFrames < maxNumFrames:
		total_reward = 0.0
		prevFeatures = np.zeros(RAMFeatures.NUM_BITS_RAM)

		while not ale.game_over() and totalNumFrames < maxNumFrames:
			# Random walk
			a = actionSet[np.random.randint(actionSet.size)]
			reward = ale.act(a);

			# Bookkeeping
			totalNumFrames += 1
			total_reward += reward

			# Obtaining real feature vector
			currFeatures = RAMFeatures.getRAMVector(ale)
			featureVector = currFeatures - prevFeatures

			# If it is a new sample, we add it to our transitions matrix
			prevSetSize  = len(setTransitions)
			setTransitions.add(tuple(featureVector))
			if len(setTransitions) != prevSetSize:
				listTransitions.append(featureVector)

			# Bookkeeping
			prevFeatures = currFeatures
			prevSetSize = len(setTransitions)

		print("Episode ended with score: " + str(total_reward))
		ale.reset_game()

	T = np.asarray(listTransitions)
	return T
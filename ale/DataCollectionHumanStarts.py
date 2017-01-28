import numpy as np

import RAMFeatures

def buildTransitionMatrix(ale, actionSet, maxNumFrames, setOfTransitions, humanTrajectory):

	# Required initializations
	total_reward = 0.0
	totalNumFrames = 0
	listTransitions = []
	prevFeatures = RAMFeatures.getRAMVector(ale)

	for i in xrange(len(humanTrajectory)):
		reward = ale.act(humanTrajectory[i]);
		total_reward += reward
		# Obtaining real feature vector
		currFeatures = RAMFeatures.getRAMVector(ale)
		featureVector = currFeatures - prevFeatures

		# If it is a new sample, we add it to our transitions matrix
		prevSetSize  = len(setOfTransitions)
		setOfTransitions.add(tuple(featureVector))
		if len(setOfTransitions) != prevSetSize:
			listTransitions.append(featureVector)

		# Bookkeeping
		prevFeatures = currFeatures
		prevSetSize = len(setOfTransitions)

	print 'Human', len(listTransitions)

	# Now I collect data on random walks:
	while not ale.game_over():
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
		prevSetSize  = len(setOfTransitions)
		setOfTransitions.add(tuple(featureVector))
		if len(setOfTransitions) != prevSetSize:
			listTransitions.append(featureVector)

		# Bookkeeping
		prevFeatures = currFeatures
		prevSetSize = len(setOfTransitions)

	print 'Random', len(listTransitions)
	print("Episode ended with score: " + str(total_reward))
	ale.reset_game()

	T = np.asarray(listTransitions)
	return setOfTransitions, T
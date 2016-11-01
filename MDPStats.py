'''
This class implements methods that give us information about the MDP such as
expected number of time steps between any two points following a policy.

Author: Marlos C. Machado
'''
import sys
import math
import numpy as np

import matplotlib.pylab as plt

from Learning import Learning

class MDPStats:

	gamma = 0.9
	numStates = 0
	actionSet = None
	environment = None

	def __init__(self, gamma, env, augmentActionSet=False):
		'''Initialize variables that are useful everywhere.'''
		self.gamma = gamma
		self.environment = env
		self.numStates = env.getNumStates() + 1


		if augmentActionSet:
			self.actionSet = np.append(env.getActionSet(), ['terminate'])
		else:
			self.actionSet = env.getActionSet()

	def _computeAvgOnMDP(self, V, ignoreZeros=True):
		''' Just average the values in a vector. One can ignore zeros.'''

		counter = 0
		summation = 0

		for i in xrange(len(V)):
			if V[i] != 0:
				summation += V[i]
				counter += 1

		return summation / counter

	def getAvgNumStepsBetweenEveryPoint(self, pi, fullActionSet, optionsActionSet, numOptionsToConsider=0, debug=False):
		''' '''
		print
		toPlot = []
		numPrimitiveActions = 4

		actionSetToUse = fullActionSet[:numPrimitiveActions]

		for i in xrange(numOptionsToConsider + 1):
			avgs = []

			# I'm going to use a matrix encoding the random policy. For each state
			# I encode the equiprobable policy for primitive actions and options
			pi = []
			for j in xrange(self.numStates):
				pi.append([])
				for k in xrange(numPrimitiveActions + i):
					pi[j].append(1.0/float(numPrimitiveActions + i))

			if i > 0:
				actionSetToUse.append(fullActionSet[numPrimitiveActions + i - 1])

			print 'Obtaining shortest paths for ' + str(numPrimitiveActions) + ' primitive actions and ' + str(i) + ' options.'
			for s in xrange(self.environment.getNumStates()):
				goalChanged = self.environment.defineGoalState(s)

				if goalChanged:
					bellman = Learning(
						self.gamma, self.environment, augmentActionSet=False)
					expectation = bellman.solveBellmanEquations(pi, actionSetToUse, optionsActionSet)

					if debug:
						for j in xrange(len(expectation) - 1):
							sys.stdout.write("%.2f\t" % (-1.0 * expectation[j]))
							if (j + 1) % 5 == 0:
								print
						print

					avgs.append(self._computeAvgOnMDP((-1.0 * expectation)))

			toPlot.append(sum(avgs) / float(len(avgs)))

		print toPlot
		if numOptionsToConsider > 0:
			plt.plot(toPlot)
			plt.show()

		return toPlot

	'''
	TODO:
		- Allow multiple options in the policy, I have to parametrize it
		- Implement the code that adds options iteratively, so I can plot a graph
		- Implement threshold for the initiation set. This should be easy to add
		- Support random behavior after the policy is over. To do so, we may
		  need to implement a monte-carlo method to estimate the probability an
		  option will terminate at each state. Then we would return (r, s', p)
		  for the Bellman equations.
	'''

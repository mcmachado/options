'''
This class implements methods that give us information about the MDP such as
expected number of time steps between any two points following a policy.

Author: Marlos C. Machado
'''
import sys
import math
import numpy as np

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

	def getAvgNumStepsBetweenEveryPoint(self, pi):
		''' '''

		avgs = []

		for s in xrange(self.environment.getNumStates()):
			goalChanged = self.environment.defineGoalState(s)

			if goalChanged:
				bellman = Learning(
					self.gamma, self.environment, augmentActionSet=False)
				expectation = bellman.solveBellmanEquations(pi)

				for i in xrange(len(expectation) - 1):
					sys.stdout.write(str(expectation[i]) + '\t')
					if (i + 1) % 5 == 0:
						print
				print
				avgs.append(self._computeAvgOnMDP(expectation))

		return sum(avgs) / float(len(avgs))

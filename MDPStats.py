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

	def getAvgNumStepsBetweenEveryPoint(self, pi):
		''' '''

		#This solution is slower and it does not work for gamma = 1
		#if self.gamma == 1:
		#	print 'This won\'t work...'
		#	sys.exit()
		#polEval = Learning(self.gamma, environment, augmentActionSet=False)
		#expectation = polEval.solvePolicyEvaluation(pi)

		bellman = Learning(self.gamma, self.environment, augmentActionSet=False)
		expectation = bellman.solveBellmanEquations(pi)

		for i in xrange(len(expectation) - 1):
			sys.stdout.write(str(expectation[i]) + '\t')
			if (i + 1) % 4 == 0:
				print
		print
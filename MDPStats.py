'''
This class implements methods that give us information about the MDP such as
expected number of time steps between any two points following a policy.

Author: Marlos C. Machado
'''
import sys
import math
import numpy as np

from Drawing import Plotter
from Learning import Learning

class MDPStats:

	gamma = 0.9
	numStates = 0
	actionSet = None
	outputPath = None
	environment = None

	def __init__(self, gamma, env, outputPath, augmentActionSet=False):
		'''Initialize variables that are useful everywhere.'''
		self.gamma = gamma
		self.environment = env
		self.numStates = env.getNumStates() + 1
		self.outputPath = outputPath


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

	def getAvgNumStepsBetweenEveryPoint(self, fullActionSet, optionsActionSet, numOptionsToConsider=0, debug=False):
		''' '''
		print
		toPlot = []
		numPrimitiveActions = 4

		actionSetToUse = fullActionSet[:numPrimitiveActions]

		for i in xrange(numOptionsToConsider + 1):
			avgs = []

			# I'm going to use a matrix encoding the random policy. For each state
			# I encode the equiprobable policy for primitive actions and options
			# However, I need to add the condition that, if the option's policy says
			# terminate, it should have probability zero for the equiprobable policy.
			pi = []
			for j in xrange(self.numStates - 1):
				pi.append([])

				for k in xrange(numPrimitiveActions):
					pi[j].append(1.0)

				if i > 0:
					for k in xrange(i): #current number of options to consider
						if optionsActionSet[i][fullActionSet[numPrimitiveActions + k][j]] == "terminate":
							pi[j].append(0.0)
						else:
							pi[j].append(1.0)

				denominator = sum(pi[j])
				for k in xrange(len(pi[j])):
					pi[j][k] = pi[j][k]/denominator

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

		if numOptionsToConsider > 0:
			plt = Plotter(self.outputPath, self.environment)
			plt.plotLine(xrange(len(toPlot)), toPlot, '# options', 'Avg. # steps',
				'Avg. # steps between any two points', 'avg_num_steps.pdf')

			#plt.plot(toPlot)
			#plt.show()

		return toPlot

	'''
	TODO:
		-- An option is not available in the termination condition
			-- To fix it is probably enough to define prob 0 for these states
		- Implement threshold for the initiation set. This should be easy to add
		- Support random behavior after the policy is over. To do so, we may
		  need to implement a monte-carlo method to estimate the probability an
		  option will terminate at each state. Then we would return (r, s', p)
		  for the Bellman equations.
	'''

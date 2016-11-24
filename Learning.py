'''
This class implements policy iteration so we can solve an MDP and extract the
optimal policy. These learned policies will compose an option to be specified.

Author: Marlos C. Machado
'''
import math
import numpy as np

class Learning:

	V = None
	pi = None
	gamma = 0.9
	numStates = 0
	actionSet = None
	environment = None

	def __init__(self, gamma, env, augmentActionSet=False):
		'''Initialize variables that are useful everywhere.'''
		self.gamma = gamma
		self.environment = env
		self.numStates = env.getNumStates() + 1

		self.V = np.zeros(self.numStates + 1)
		self.pi = np.zeros(self.numStates + 1, dtype = np.int)

		if augmentActionSet:
			self.actionSet = np.append(env.getActionSet(), ['terminate'])
		else:
			self.actionSet = env.getActionSet()

	def _evalPolicy(self):
		''' Policy evaluation step.'''
		delta = 0.0
		for s in xrange(self.numStates):
			v = self.V[s]
			nextS, nextR = self.environment.getNextStateAndReward(
				s, self.actionSet[self.pi[s]])

			self.V[s] = nextR + self.gamma * self.V[nextS]
			delta = max(delta, math.fabs(v - self.V[s]))

		return delta

	def _improvePolicy(self):
		''' Policy improvement step. '''
		policy_stable = True
		for s in xrange(self.numStates):
			old_action = self.pi[s]
			tempV = [0.0] * len(self.actionSet)
			# I first get all value-function estimates
			for i in xrange(len(self.actionSet)):
				nextS, nextR = self.environment.getNextStateAndReward(
					s, self.actionSet[i])
				tempV[i] = nextR + self.gamma * self.V[nextS]

			# Now I take the argmax
			self.pi[s] = np.argmax(tempV)
			# I break ties always choosing to terminate:
			if math.fabs(tempV[self.pi[s]] - tempV[(len(self.actionSet) - 1)]) < 0.001:
				self.pi[s] = (len(self.actionSet) - 1)
			if old_action != self.pi[s]:
				policy_stable = False

		return policy_stable

	def solvePolicyIteration(self, theta=0.001):
		''' Implementation of Policy Iteration, as in the policy iteration
		    pseudo-code presented in Sutton and Barto (2016).'''

		# Initialization is done in the constructor
		policy_stable = False

		while not policy_stable:
			# Policy evaluation
			delta = self._evalPolicy()
			while (theta < delta):
				delta = self._evalPolicy()

			# Policy improvement
			policy_stable = self._improvePolicy()

		return self.V, self.pi

	def solvePolicyEvaluation(self, pi, theta=0.001):
		'''Implementation of Policy Evaluation, as in the policy evaluation
		   pseudo-code presented in Sutton and Barto (2016).'''

		# I'll use the same V, it shouldn't really matter,
		# although ideally these things should be independent
		self.V = np.zeros(self.numStates + 1)
		iteration = 1

		delta = 1
		while delta > theta:
			delta = 0
			for s in xrange(self.numStates - 1):
				v = self.V[s]
				tempSum = 0
				for a in xrange(len(pi[s])):
					nextS, nextR = self.environment.getNextStateAndReward(
						s, self.actionSet[a])
					tempSum += pi[s][a] * 1.0 * (
						nextR + self.gamma * self.V[nextS])

				self.V[s] = tempSum
				delta = max(delta, math.fabs(v - self.V[s]))

			if iteration %1000 == 0:
				print 'Iteration:', iteration, '\tDelta:', delta
			iteration += 1

		'''
		import sys
		for i in xrange(16):
			sys.stdout.write(str(self.V[i]) + ' ')
			if (i + 1) % 4 == 0:
				print
		'''
		return self.V

	def solveBellmanEquations(self, pi, fullActionSet, optionsActionSet):
		''' This method generates the Bellman equations using the model
			available in self.environment and solves the generated set of
			linear equations.'''

		numberOfPrimitiveActions = 4
		# ax = b
		a_equations = np.zeros((self.numStates, self.numStates))
		b_equations = np.zeros(self.numStates)

		'''
		# V[s] = \sum \pi(a|s) \sum p(s',r|s,a) [r + \gamma V[s']]
		# V[s] = \sum \pi(a|s) 1.0 [r + \gamma V[s']] (assuming determinism)
		# - \sum \pi(a|s) r = -V[s] + \sum \pi(a|s) \gamma V[s']
		'''
		for s in xrange(self.numStates - 1):
			a_equations[s][s] = -1
			for a in xrange(len(pi[s])):
				nextS = -1
				nextR = -1

				#If it is a primitive action
				if isinstance(fullActionSet[a], basestring):
					nextS, nextR = self.environment.getNextStateAndReward(
						s, fullActionSet[a])
				else: #if it is an option
					nextS, nextR = self.environment.getNextStateAndRewardFromOption(
						s, fullActionSet[a],
						optionsActionSet[a - numberOfPrimitiveActions])

				a_equations[s][nextS] += pi[s][a] * self.gamma
				b_equations[s] -= pi[s][a] * nextR

		for i in xrange(self.numStates):
			hasOnlyZeros = True
			for j in xrange(self.numStates):
				if a_equations[i][j] != 0.0:
					hasOnlyZeros = False

			if hasOnlyZeros:
				a_equations[i][i] = 1
				b_equations[i] = 0


		expectation = np.linalg.solve(a_equations, b_equations)
		return expectation

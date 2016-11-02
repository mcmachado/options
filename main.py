'''
Main file. From here I call all the relevant functions that allow me to test my
algorithm, including obtaining the graph Laplacian, learning an optimal policy
given a reward function, and plotting options and basis functions.

Author: Marlos C. Machado
'''
import sys
import numpy as np
import argparse

from Learning import Learning
from Drawing import Plotter
from Utils import Utils
from Environment import GridWorld
from MDPStats import MDPStats

def discoverOptions(env, plotGraphs=True):
	#I'll need this when computing the expected number of steps:
	options = []
	actionSetPerOption = []

	# Computing the Combinatorial Laplacian
	W = env.getAdjacencyMatrix()
	D = np.zeros((numStates, numStates))

	# Obtaining the Valency Matrix
	for i in xrange(numStates):
		for j in xrange(numStates):
			D[i][i] = np.sum(W[i])
	# Making sure our final matrix will be full rank
	for i in xrange(numStates):
	   if D[i][i] == 0.0:
	       D[i][i] = 1.0

	# Normalized Laplacian
	L = D - W
	expD = Utils.exponentiate(D, -0.5)
	normalizedL = expD.dot(L).dot(expD)

	# Eigendecomposition
	eigenvalues, eigenvectors = np.linalg.eig(normalizedL)
	# I need to sort the eigenvalues and eigenvectors
	idx = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]

	if plotGraphs:
		# Plotting all the basis
		plot = Plotter(outputPath, env)
		plot.plotBasisFunctions(eigenvalues, eigenvectors)

	# Now I will define a reward function and solve the MDP for it
	guard = len(eigenvectors)
	for i in xrange(guard):
		idx = guard - i - 1
		print 'Solving for eigenvector #' + str(idx)
		polIter = Learning(0.9, env, augmentActionSet=True)
		env.defineRewardFunction(eigenvectors[:,idx])
		V, pi = polIter.solvePolicyIteration()

		if plotGraphs:
			plot.plotValueFunction(V[0:numStates], str(idx) + '_')
			plot.plotPolicy(pi[0:numStates], str(idx) + '_')

		options.append(pi[0:numStates])
		optionsActionSet = env.getActionSet()
		optionsActionSet.append('terminate')
		actionSetPerOption.append(optionsActionSet)

	env.defineRewardFunction(None) #I need to do this now that I'm done with the PVFs
	env.resetEnvironment()

	return options, actionSetPerOption

def testPolicyEvaluation(env):
	''' Simple test for policy evaluation '''

	pi = numStates * [[0.25, 0.25, 0.25, 0.25]]

	#This solution is slower and it does not work for gamma = 1
	#polEval = Learning(0.9999, env, augmentActionSet=False)
	#expectation = polEval.solvePolicyEvaluation(pi)

	bellman = Learning(1, env, augmentActionSet=False)
	expectation = bellman.solveBellmanEquations(pi)

	for i in xrange(len(expectation) - 1):
		sys.stdout.write(str(expectation[i]) + '\t')
		if (i + 1) % 4 == 0:
			print
	print

def testPolicyIteration(env):
	''' Simple test for policy iteration '''

	polIter = Learning(0.9, env, augmentActionSet=False)
	V, pi = polIter.solvePolicyIteration()

	# I'll assign the goal as the termination action
	pi[env.getGoalState()] = 4

	# Now we just plot the learned value function and the obtained policy
	plot = Plotter(outputPath, env)
	plot.plotValueFunction(V[0:numStates], 'goal_')
	plot.plotPolicy(pi[0:numStates], 'goal_')

def testOptionDiscoveryThroughPVFs(env):
	''' Simple test for option discovery through proto-value functions. '''
	options, actionSetPerOption = discoverOptions(env, plotGraphs=True)

def getExpectedNumberOfStepsFromOption(env):

	# We first discover all options
	actionSet = env.getActionSet()
	options, actionSetPerOption = discoverOptions(env)

	# Now I add all options to my action set. Later we decide which ones to use.
	for i in xrange(len(options)):
		actionSet.append(options[i])

	return stats.getAvgNumStepsBetweenEveryPoint(actionSet, actionSetPerOption, numOptionsToConsider=10)


if __name__ == "__main__":
	# Parse command line
	parser = argparse.ArgumentParser(
		description='Obtain proto-value functions, options, graphs, etc.')
	parser.add_argument('-i', '--input', type = str, default = 'mdps/fig1.mdp',
		help='File containing the MDP definition.')
	parser.add_argument('-o', '--output', type = str, default = 'graphs/fig1_',
		help='Prefix that will be used to generate all outputs.')

	args = parser.parse_args()

	inputMDP = args.input
	outputPath = args.output

	# Create environment
	env = GridWorld(path = inputMDP)
	numStates = env.getNumStates()
	numRows, numCols = env.getGridDimensions()

	#testOptionDiscoveryThroughPVFs(env)
	#testPolicyIteration(env)
	#testPolicyEvaluation(env)

	gamma = 1.0
	pi = numStates * [[0.25, 0.25, 0.25, 0.25]]

	stats = MDPStats(gamma, env)
	#print stats.getAvgNumStepsBetweenEveryPoint(pi, env.getActionSet(),  None)
	getExpectedNumberOfStepsFromOption(env)

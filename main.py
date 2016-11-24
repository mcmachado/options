'''
Main file. From here I call all the relevant functions that allow me to test my
algorithm, including obtaining the graph Laplacian, learning an optimal policy
given a reward function, and plotting options and basis functions.

Author: Marlos C. Machado
'''
import sys
import warnings
import numpy as np

from Learning import Learning
from Drawing import Plotter
from Utils import Utils
from Utils import ArgsParser
from Environment import GridWorld
from MDPStats import MDPStats

def discoverOptions(env, epsilon, verbose, discoverNegation, plotGraphs=False):
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
	# IMPORTANT: The eigenvectors are in columns
	eigenvalues, eigenvectors = np.linalg.eig(normalizedL)
	# I need to sort the eigenvalues and eigenvectors
	idx = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]

	# If I decide to use both directions of the eigenvector, I do it here.
	# It is easier to just change the list eigenvector, even though it may
	# not be the most efficient solution. The rest of the code remains the same.
	if discoverNegation:
		oldEigenvalues = eigenvalues
		oldEigenvectors = eigenvectors.T
		eigenvalues = []
		eigenvectors = []
		for i in xrange(len(oldEigenvectors)):
			eigenvalues.append(oldEigenvalues[i])
			eigenvalues.append(oldEigenvalues[i])
			eigenvectors.append(oldEigenvectors[i])
			eigenvectors.append(-1 * oldEigenvectors[i])

		eigenvalues = np.asarray(eigenvalues)
		eigenvectors = np.asarray(eigenvectors).T

	if plotGraphs:
		# Plotting all the basis
		plot = Plotter(outputPath, env)
		plot.plotBasisFunctions(eigenvalues, eigenvectors)

	# Now I will define a reward function and solve the MDP for it
	# I iterate over the columns, not rows. I can index by 0 here.
	guard = len(eigenvectors[0])
	for i in xrange(guard):
		idx = guard - i - 1
		if verbose:
			print 'Solving for eigenvector #' + str(idx)
		polIter = Learning(0.9, env, augmentActionSet=True)
		env.defineRewardFunction(eigenvectors[:,idx])
		V, pi = polIter.solvePolicyIteration()

		# Now I will eliminate any actions that may give us a small improvement.
		# This is where the epsilon parameter is important. If it is not set all
		# it will never be considered, since I set it to a very small value
		for j in xrange(len(V)):
			if V[j] < epsilon:
				pi[j] = len(env.getActionSet())

		if plotGraphs:
			plot.plotValueFunction(V[0:numStates], str(idx) + '_')
			plot.plotPolicy(pi[0:numStates], str(idx) + '_')

		options.append(pi[0:numStates])
		optionsActionSet = env.getActionSet()
		optionsActionSet.append('terminate')
		actionSetPerOption.append(optionsActionSet)

	#I need to do this after I'm done with the PVFs:
	env.defineRewardFunction(None)
	env.resetEnvironment()

	return options, actionSetPerOption

def policyEvaluation(env):
	''' Simple test for policy evaluation '''

	pi = numStates * [[0.25, 0.25, 0.25, 0.25]]
	actionSet = env.getActionSet()

	#This solution is slower and it does not work for gamma = 1
	#polEval = Learning(0.9999, env, augmentActionSet=False)
	#expectation = polEval.solvePolicyEvaluation(pi)

	bellman = Learning(1, env, augmentActionSet=False)
	expectation = bellman.solveBellmanEquations(pi, actionSet, None)

	for i in xrange(len(expectation) - 1):
		sys.stdout.write(str(expectation[i]) + '\t')
		if (i + 1) % env.numCols == 0:
			print
	print

def policyIteration(env):
	''' Simple test for policy iteration '''

	polIter = Learning(0.9, env, augmentActionSet=False)
	V, pi = polIter.solvePolicyIteration()

	# I'll assign the goal as the termination action
	pi[env.getGoalState()] = 4

	# Now we just plot the learned value function and the obtained policy
	plot = Plotter(outputPath, env)
	plot.plotValueFunction(V[0:numStates], 'goal_')
	plot.plotPolicy(pi[0:numStates], 'goal_')

def optionDiscoveryThroughPVFs(env, epsilon, verbose, discoverNegation):
	''' Simple test for option discovery through proto-value functions. '''
	options, actionSetPerOption = discoverOptions(env,
		epsilon=epsilon, verbose=verbose,
		discoverNegation=discoverNegation, plotGraphs=True)

def getExpectedNumberOfStepsFromOption(env, eps, verbose,
	discoverNegation, loadedOptions=None):

	# We first discover all options
	options = None
	actionSetPerOption = None
	actionSet = env.getActionSet()

	if loadedOptions == None:
		if verbose:
			options, actionSetPerOption = discoverOptions(env, eps, verbose,
				discoverNegation, plotGraphs=True)
		else:
			options, actionSetPerOption = discoverOptions(env, eps, verbose,
				discoverNegation, plotGraphs=False)
	else:
		options = loadedOptions
		actionSetPerOption = []
		for i in xrange(len(loadedOptions)):
			tempActionSet = env.getActionSet()
			tempActionSet.append('terminate')
			actionSetPerOption.append(tempActionSet)

	# Now I add all options to my action set. Later we decide which ones to use.
	for i in xrange(len(options)):
		actionSet.append(options[i])

	if discoverNegation:
		numOptions = 2*env.getNumStates()
	else:
		numOptions = env.getNumStates()

	if discoverNegation:
		for i in xrange(numOptions/2):
			listToPrint = stats.getAvgNumStepsBetweenEveryPoint(actionSet,
				actionSetPerOption, verbose, initOption=i*2,
				numOptionsToConsider=2)
			myFormattedList = [ '%.2f' % elem for elem in listToPrint ]
			print 'Random, Option ' + str(i + 1) + ': ' + str(myFormattedList)
	else:
		for i in xrange(numOptions):
			listToPrint = stats.getAvgNumStepsBetweenEveryPoint(actionSet,
				actionSetPerOption, verbose, initOption=i,
				numOptionsToConsider=1)
			myFormattedList = [ '%.2f' % elem for elem in listToPrint ]
			print 'Random, Option ' + str(i + 1) + ': ' + str(myFormattedList)

	print 'Average number of steps between any two states as we add options:'
	listToPrint = stats.getAvgNumStepsBetweenEveryPoint(actionSet,
		actionSetPerOption, verbose, initOption=0,
		numOptionsToConsider=numOptions)
	myFormattedList = [ '%.2f' % elem for elem in listToPrint ]
	print myFormattedList

if __name__ == "__main__":

	#Read input arguments
	args = ArgsParser.readInputArgs()

	taskToPerform = args.task
	epsilon = args.epsilon
	verbose = args.verbose
	inputMDP = args.input
	outputPath = args.output
	optionsToLoad = args.load
	bothDirections = args.both

	if not verbose:
		warnings.filterwarnings('ignore')

	# Create environment
	env = GridWorld(path = inputMDP)
	numStates = env.getNumStates()
	numRows, numCols = env.getGridDimensions()

	# I may load options if I'm told so:
	loadedOptions = None
	if optionsToLoad != None:
		loadedOptions = []
		for i in xrange(len(optionsToLoad)):
			loadedOptions.append(Utils.loadOption(optionsToLoad[i]))
			plot = Plotter(outputPath, env)
			plot.plotPolicy(loadedOptions[i], str(i+1) + '_')

	if taskToPerform == 1:
		optionDiscoveryThroughPVFs(env, epsilon=epsilon, verbose=verbose,
			discoverNegation=bothDirections)
	elif taskToPerform == 2:
		policyIteration(env)
	elif taskToPerform == 3:
		#TODO: I should allow one to evaluate a loaded policy
		policyEvaluation(env)
	elif taskToPerform == 4:
		gamma = 1.0
		stats = MDPStats(gamma, env, outputPath)
		getExpectedNumberOfStepsFromOption(env, epsilon, verbose,
			bothDirections, loadedOptions=loadedOptions)

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

	# Plotting all the basis
	plot = Plotter(outputPath, env)
	
	plot.plotBasisFunctions(eigenvalues, eigenvectors)
	# Now I will define a reward function and solve the MDP for it
	guard = len(eigenvectors)
	for i in xrange(guard):
		idx = guard - i - 1
		print 'Solving for eigenvector #' + str(idx)
		polIter = Learning(0.9, env)
		env.defineRewardFunction(eigenvectors[:,idx])
		V, pi = polIter.solvePolicyIteration()
		for i in xrange(len(V)):
			if V[i] < 0:
				pi[i] = -1
		plot.plotValueFunction(V, str(idx) + '_')
		plot.plotPolicy(pi, str(idx) + '_')

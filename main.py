import sys
import numpy as np
import matplotlib
import argparse

import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from matplotlib import cm

from Utils import Utils
from Environment import GridWorld

parser = argparse.ArgumentParser(description='Obtain proto-value functions, options, graphs, etc.')
parser.add_argument('-i', '--input', type=str, default='mdps/fig1.mdp', help='File containing the MDP definition.')
parser.add_argument('-o', '--output', type=str, default='graphs/fig1_', help='Prefix that will be used to generate all outputs.')

args = parser.parse_args()

inputMDP = args.input
outputPath = args.output

env = GridWorld(path=inputMDP)

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
for i in xrange(len(eigenvalues)):
	print eigenvalues[i]
	fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
	X, Y = np.meshgrid(np.arange(numRows), np.arange(numCols))
	Z = eigenvectors[:,i].reshape(numRows,numCols)
	my_col = cm.jet(np.random.rand(Z.shape[0],Z.shape[1]))

	ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'))
	plt.savefig(outputPath + 'eig_' + str(i) + '.png')
	plt.close()


plt.plot(eigenvalues, 'o')
plt.savefig(outputPath + 'eigenvalues.png')
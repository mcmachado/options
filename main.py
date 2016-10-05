import sys
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from matplotlib import cm
from Environment import GridWorld

def exponentiate(M, exp):
	numRows = len(M)
	numCols = len(M[0])
	expM = np.zeros((numRows, numCols))

	for i in xrange(numRows):
		for j in xrange(numCols):
			if M[i][j] != 0:
				expM[i][j] = M[i][j]**exp

	return expM

env = GridWorld(path='mdps/fig1.mdp')

numStates = env.getNumStates()
numRows, numCols = env.getGridDimensions()

# Computing the Combinatorial Laplacian
W = env.getAdjacencyMatrix()
D = np.zeros((numStates, numStates))

for i in xrange(numStates):
	for j in xrange(numStates):
		D[i][i] = np.sum(W[i])

for i in xrange(numStates):
   if D[i][i] == 0.0:
       D[i][i] = 1.0

L = D - W
expD = exponentiate(D, -0.5)
normalizedL = expD.dot(L).dot(expD)

# Computing the eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(normalizedL)
# If need to sort the eigenvalues and eigenvectors
idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

print eigenvalues

# Plotting all the basis
for i in xrange(len(eigenvalues)):
	print eigenvalues[i]
	if eigenvalues[i] > 0.0:
		fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
		X, Y = np.meshgrid(np.arange(numRows), np.arange(numCols))
		Z = eigenvectors[:,i].reshape(numRows,numCols)
		my_col = cm.jet(np.random.rand(Z.shape[0],Z.shape[1]))

		ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'))
		plt.savefig('graphs/normalized/eig_' + str(i) + '.png')
		plt.close()


plt.plot(eigenvalues, 'o')
plt.savefig('graphs/normalized/eigenvalues.png')
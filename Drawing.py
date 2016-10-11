'''
This class plots the basis functions as 3d meshes and options as arrows
projected onto the original MDP.

Author: Marlos C. Machado
'''
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.axes3d as axes3d

from matplotlib import cm

class Plotter:
	env = None
	outputPath = ''
	numRows = 0
	numCols = 0
	matrixMDP = None

	def __init__(self, output, environment):
		'''Initialize variables that will be useful everywhere.'''
		self.env = environment
		self.outputPath = output
		self.numRows, self.numCols = self.env.getGridDimensions()
		self.matrixMDP = self.env.matrixMDP

	def plotBasisFunctions(self, eigenvalues, eigenvectors):
		'''3d plot of the basis function. Right now I am plotting eigenvectors,
		   so each coordinate of the eigenvector correspond to the value to be
		   plotted for the correspondent state.''' 
		for i in xrange(len(eigenvalues)):	
			fig, ax = plt.subplots(subplot_kw = dict(projection = '3d'))
			X, Y = np.meshgrid(np.arange(self.numRows), np.arange(self.numCols))
			Z = eigenvectors[:,i].reshape(self.numRows, self.numCols)
			my_col = cm.jet(np.random.rand(Z.shape[0],Z.shape[1]))

			ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1,
				cmap = plt.get_cmap('jet'))
			plt.savefig(self.outputPath + str(i) + '_eig' + '.png')
			plt.close()


		plt.plot(eigenvalues, 'o')
		plt.savefig(self.outputPath + 'eigenvalues.png')

	def plotValueFunction(self, valueFunction, prefix):
		'''3d plot of a value function.''' 
		fig, ax = plt.subplots(subplot_kw = dict(projection = '3d'))
		X, Y = np.meshgrid(np.arange(self.numRows), np.arange(self.numCols))
		Z = valueFunction.reshape(self.numRows, self.numCols)
		my_col = cm.jet(np.random.rand(Z.shape[0],Z.shape[1]))

		ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1,
			cmap = plt.get_cmap('jet'))
		plt.savefig(self.outputPath + prefix + 'value_function.png')
		plt.close()


	def plotPolicy(self, policy, prefix):
		plt.clf()
		for idx in xrange(len(policy)):
			j = idx % self.numCols
			i = (idx - j)/self.numCols

			idx = j + i * self.numCols
			dx = 0
			dy = 0
			if policy[idx] == 0: # up
				dy = 0.35
			elif policy[idx] == 1: #right
				dx = 0.35
			elif policy[idx] == 2: #down
				dy = -0.35
			elif policy[idx] == 3: #left
				dx = -0.35
			elif self.matrixMDP[i][j] != -1 and policy[idx] == 4: # termination
				circle = plt.Circle((j + 0.5, i + 0.5), 0.025, color='k')
				plt.gca().add_artist(circle)

			if self.matrixMDP[i][j] != -1:
				plt.arrow(j + 0.5, i + 0.5, dx, dy, head_width=0.05, head_length=0.05, fc='k', ec='k')
			else:
				plt.gca().add_patch(
					patches.Rectangle(
					(j, i),   # (x,y)
					1.0,          # width
					1.0,          # height
					facecolor = "gray"
					)
				)

		plt.xlim([0, self.numCols])
		plt.ylim([0, self.numRows])
		

		for i in xrange(self.numCols):
			plt.axvline(i, color='k', linestyle=':')
		plt.axvline(self.numCols, color='k', linestyle=':')
		
		for j in xrange(self.numRows):
			plt.axhline(j, color='k', linestyle=':')
		plt.axhline(self.numRows, color='k', linestyle=':')

		plt.savefig(self.outputPath + prefix + 'policy.png')
		plt.close()

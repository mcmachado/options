'''
This class plots the basis functions as 3d meshes and options as arrows
projected onto the original MDP.

Author: Marlos C. Machado
'''
import numpy as np
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from matplotlib import cm

class Plotter:
	outputPath = ''
	numRows = 0
	numCols = 0

	def __init__(self, output, nRows, nCols):
		'''Initialize variables that will be useful everywhere.'''
		self.outputPath = output
		self.numRows = nRows
		self.numCols = nCols

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
			plt.savefig(self.outputPath + 'eig_' + str(i) + '.png')
			plt.close()


		plt.plot(eigenvalues, 'o')
		plt.savefig(self.outputPath + 'eigenvalues.png')

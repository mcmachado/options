'''
This class implements simple methods that are useful in several different places
but they are not more than simple procedures.

Author: Marlos C. Machado
'''
import argparse
import numpy as np

class Utils:
	colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

	'''Exponentiate a matrix elementwise. This is useful when you have a
	   diagonal matrix, because the exponentiation makes sense, it is equivalent
	   to exponentiating a whole matrix.'''
	@staticmethod
	def exponentiate(M, exp):
		numRows = len(M)
		numCols = len(M[0])
		expM = np.zeros((numRows, numCols))

		for i in xrange(numRows):
			for j in xrange(numCols):
				if M[i][j] != 0:
					expM[i][j] = M[i][j]**exp

		return expM

	@staticmethod
	def _readFile(path):
		''' We just read the file and put its contents in fstr.'''
		fstr = ''
		file = open(path, 'r')
		for line in file:
			fstr += line

		return fstr

	@staticmethod
	def loadOption(filePath):
		''' I now parse the received string. I'll store everything in a matrix
		(matrixMDP) such that U, D, R, L, T mean 'up', 'down', 'right', 'left',
		'terminate', referring to the proper action to be taken.'''
		fstr = Utils._readFile(filePath)
		data = fstr.split('\n')
		numRows = int(data[0].split(',')[0])
		numCols = int(data[0].split(',')[1])
		matrixMDP = np.zeros((numRows, numCols), dtype = np.int)

		for i in xrange(len(data) - 1):
			for j in xrange(len(data[i+1])):
				if data[i+1][j] == 'X':
					matrixMDP[i][j] = 4 #terminate
				elif data[i+1][j] == 'T':
					matrixMDP[i][j] = 4 #terminate
				elif data[i+1][j] == 'L':
					matrixMDP[i][j] = 3 #left
				elif data[i+1][j] == 'D':
					matrixMDP[i][j] = 2 #down
				elif data[i+1][j] == 'R':
					matrixMDP[i][j] = 1 #right
				elif data[i+1][j] == 'U':
					matrixMDP[i][j] = 0 #up


		option = []
		for i in xrange(numRows):
			for j in xrange(numCols):
				option.append(matrixMDP[i][j])

		return option

	@staticmethod
	def movingAverage(data, n=50):
		ret = np.cumsum(data, dtype=float)
		ret[n:] = ret[n:] - ret[:-n]
		return ret[n - 1:] / n

	@staticmethod
	def computeConfInterval(avg, std_dev, n):
		return avg - 1.96 * (std_dev/np.sqrt(n)), avg + 1.96 * (std_dev/np.sqrt(n))

class ArgsParser:
	'''Read the user's input and parse the arguments properly. When returning
	args, each value is properly filled. Ideally one shouldn't have to read
	this function to access the proper arguments, but I postone this.'''
	@staticmethod
	def readInputArgs():
		# Parse command line
		parser = argparse.ArgumentParser(
			description='Obtain proto-value functions, options, graphs, etc.')

		parser.add_argument('-t', '--task', type = int, default = 1,
			help='Task to be performed (default: 1). ' +
			'1: Discover options; ' +
			'2: Solve for a given goal (policy iteration); ' +
			'3: Evaluate random policy (policy evaluation); ' +
			'4: Compute the average number of time steps between any two states;' +
			'5: Solve for a given goal (q-learning);' +
			'6: Solve for a given goal w/ primitive actions (q-learning)' +
			' following options;' +
			'7: Solve for a given goal w/ primitive actions (q-learning)' +
			' following discovered AND loaded options This one is for comparison.')

		parser.add_argument('-i', '--input', type = str, default = 'mdps/toy.mdp',
			help='File containing the MDP definition (default: mdps/toy.mdp).')

		parser.add_argument('-o', '--output', type = str, default = 'graphs/',
			help='Prefix that will be used to generate all outputs (default: graphs/).')

		parser.add_argument('-l', '--load', type = str, nargs = '+', default = None,
			help='List of files that contain the options to be loaded (default: None).')

		parser.add_argument('-e', '--epsilon', type = float, default = 0,
			help='Epsilon threshold to define options\' termination condition (default: 0).')

		parser.add_argument('-b', '--both', action='store_true', default = False,
			help='When discovering options, we should use both directions of ' +
			'the eigenvectors (positive and negative) (default: False).')

		parser.add_argument('-v', '--verbose', action='store_true', default=False,
			help='Verbose output. If not set, intermediate printing information' +
			' is suppressed. When using this option with -v=4, no graphs are ' +
			' plotted (default: False).')

		parser.add_argument('-s', '--num_seeds', type = int, default = 5,
			help='Number of seeds to be averaged over when appropriate (default: 5).')

		parser.add_argument('-m', '--max_length_ep', type = int, default = 100,
			help='Maximum number of time steps an episode may last (default: 100).')

		parser.add_argument('-n', '--num_episodes', type = int, default = 1000,
			help='Number of episodes in which learning will happen (default: 1000).')

		args = parser.parse_args()

		return args

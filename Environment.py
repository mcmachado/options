'''
This class implements simple gridworlds. It reads a string, or a file containing
the string, and generates the MDP. I just implemented a tabular representation
for this class, it is hard to try to propose something else for gridworlds.

Author: Marlos C. Machado
'''
import sys
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

from matplotlib import cm

np.set_printoptions(threshold=np.inf)

class GridWorld:
	
	strMDP  = ''
	numRows = -1
	numCols = -1
	numStates = -1
	matrixMDP = None
	adjMatrix = None
	rewardFunction = None

	currX = -1
	currY = -1
	startX = -1
	startY = -1

	def __init__(self, path=None, strin=None):
		'''Return a GridWorld object that instantiates the MDP defined in a file
		(specified in path). In case it is None, then the MDP definition is read
		from strin, which is a string with the content that path would hold. The
		input should have a very specific format. The first line should contain
		two numbers separated by a comma. These numbers define the dimensions of
		the MDP. The rest of the lines are composed of X's denoting walls and of
		.'s denoting empty spaces in the MDP. S denotes the starting state.'''
		if path != None:
			self._readFile(path)
		elif strin != None:
			self.strMDP = strin
		else:
			print 'You are supposed to provide an MDP specification as input!'
			sys.exit()

		self._parseString()

		self.currX = self.startX
		self.currY = self.startY
		self.numStates = self.numRows * self.numCols

	def _readFile(self, path):
		''' We just read the file and put its contents in strMDP.'''
		file = open(path, 'r')
		for line in file:
			self.strMDP += line

	def _parseString(self):
		''' I now parse the received string. I'll store everything in a matrix
		(matrixMDP) such that -1 means wall and 0 means available square. The
		letter 'S' is converted to the initial (x,y) position. '''
		data = self.strMDP.split('\n')
		self.numRows = int(data[0].split(',')[0])
		self.numCols = int(data[0].split(',')[1])
		self.matrixMDP = np.zeros((self.numRows, self.numCols))

		for i in xrange(len(data) - 1):
			for j in xrange(len(data[i+1])):
				if data[i+1][j] == 'X':
					self.matrixMDP[i][j] = -1
				elif data[i+1][j] == '.':
					self.matrixMDP[i][j] = 0
				elif data[i+1][j] == 'S':
					self.startX = i
					self.startY = j

	def _getNextState(self, action):
		if action == 'up':
			return self.currX - 1, self.currY
		elif action == 'right':
			return self.currX, self.currY + 1
		elif action == 'down':
			return self.currX + 1, self.currY
		elif action == 'left':
			return self.currX, self.currY - 1

	def act(self, action):
		''' At first there are four possible actions: up, down, left and right.
		If the agent tries to go to a -1 state it will stay on the same coord.
		I decided to not implement any stochasticity for now.'''

		# Basically I get what will be the next state and before really making
		# it my current state I verify everything is sound.
		# TODO: Implement rewards
		nextX, nextY = self._getNextState(action)
		if nextX < 0 or nextY < 0:
			print 'You were supposed to have hit a wall before!' 
			print 'There is something wrong with your MDP definition.'
			sys.exit()

		if nextY == len(self.matrixMDP) or nextX == len(self.matrixMDP[nextY]):
			print 'You were supposed to have hit a wall before!' 
			print 'There is something wrong with your MDP definition.'
			sys.exit()

		if self.matrixMDP[nextX][nextY] != -1:
			self.currX = nextX
			self.currY = nextY

	def getGridDimensions(self):
		return self.numRows, self.numCols

	def getNumStates(self):
		return self.numStates

	def getActionSet(self):
		''' At first the four directional actions are the ones available.'''
		return ['up', 'right', 'down', 'left']

	def _fillAdjacencyMatrix(self):
		''' This is not efficient, but for small MDPs it should be fast.'''
		self.adjMatrix = np.zeros((self.numStates, self.numStates), dtype = np.int)
		self.idxMatrix = np.zeros((self.numRows, self.numCols), dtype = np.int)

		'''I'll try for all states not in the borders (they have to be walls)
		all 4 possible directions. If the next state is also available we add
		such entry to the adjancency matrix, otherwise we don't.'''
		for i in xrange(len(self.idxMatrix)):
			for j in xrange(len(self.idxMatrix[i])):
				self.idxMatrix[i][j] = i * self.numCols + j

		for i in xrange(len(self.matrixMDP)):
			for j in xrange(len(self.matrixMDP[i])):
				if i != 0 and i != (self.numRows - 1) and j != 0 and j != (self.numCols - 1):
					if self.matrixMDP[i + 1][j] != -1:
						self.adjMatrix[self.idxMatrix[i][j]][self.idxMatrix[i + 1][j]] = 1
					if self.matrixMDP[i - 1][j] != -1:
						self.adjMatrix[self.idxMatrix[i][j]][self.idxMatrix[i - 1][j]] = 1
					if self.matrixMDP[i][j + 1] != -1:
						self.adjMatrix[self.idxMatrix[i][j]][self.idxMatrix[i][j + 1]] = 1
					if self.matrixMDP[i][j - 1] != -1:
						self.adjMatrix[self.idxMatrix[i][j]][self.idxMatrix[i][j - 1]] = 1

	def getAdjacencyMatrix(self):
		''' If I never did it before, I will fill the adjacency matrix.
		Otherwise I'll just return the one that was filled before.'''
		if self.adjMatrix == None:
			self._fillAdjacencyMatrix()

		return self.adjMatrix

	def defineRewardFunction(self, vector):
		self.rewardFunction = vector
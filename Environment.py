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

	currX = 0
	currY = 0
	startX = 0
	startY = 0
	goalX = 0
	goalY = 0

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
					self.matrixMDP[i][j] = 0
					self.startX = i
					self.startY = j
				elif data[i+1][j] == 'G':
					self.matrixMDP[i][j] = 0
					self.goalX = i
					self.goalY = j

	def _getStateIndex(self, x, y):
		''' Given a state coordinate (x,y) this method returns the index that
			uniquely identifies this state.'''
		idx = y + x * self.numCols
		return idx

	def getStateXY(self, idx):
		''' Given the index that uniquely identifies each state this method
			returns its equivalent coordinate (x,y).'''
		y = idx % self.numCols
		x = (idx - y)/self.numCols

		return x, y

	def _getNextState(self, action):
		''' This function returns what is going to be the next state (x,y)
		    given an action. It does not update the next state, it is a one-step
		    forward model. '''

		nextX = self.currX
		nextY = self.currY

		if action == 'terminate':
			# In this case we are not discovering options
			# we are just talking about a general MDP.
			if self.rewardFunction == None:
				if nextX == self.goalX and nextY == self.goalY:
					return -1, -1 # absorbing state
				else:
					return self.currX, self.currY
			# Otherwise we are talking about option discovery,
			# so when an option terminates it should stop "suffering".
			else:
				return -1, -1 # absorbing state

		if self.matrixMDP[self.currX][self.currY] != -1:
			if action == 'up' and self.currX > 0:
				nextX = self.currX - 1
				nextY = self.currY
			elif action == 'right' and self.currY < self.numCols - 1:
				nextX = self.currX
				nextY = self.currY + 1
			elif action == 'down' and self.currX < self.numRows - 1:
				nextX = self.currX + 1
				nextY = self.currY
			elif action == 'left' and self.currY > 0:
				nextX = self.currX
				nextY = self.currY - 1

		if nextX < 0 or nextY < 0:
			print 'You were supposed to have hit a wall before!' 
			print 'There is something wrong with your MDP definition.'
			sys.exit()

		if nextX == len(self.matrixMDP) or nextY == len(self.matrixMDP[0]):
			print 'You were supposed to have hit a wall before!' 
			print 'There is something wrong with your MDP definition.'
			sys.exit()

		if self.matrixMDP[nextX][nextY] != -1:
			return nextX, nextY
		else:
			return self.currX, self.currY

	def getCurrentState(self):
		''' Returns the unique identifier for the current state the agent is.'''

		currStateIdx = self._getStateIndex(self.currX, self.currY)
		return currStateIdx

	def getGoalState(self):
		''' Returns the unique identifier to the goal.'''
		goalStateIdx = self._getStateIndex(self.goalX, self.goalY)
		return goalStateIdx

	def _getNextReward(self, currX, currY, action, nextX, nextY):
		''' Returns the reward the agent will observe if in state (currX, currY)
			and it takes action 'action' leading to the state (nextX, nextY).'''

		# If a reward vector was not informed we get -1 everywhere until
		# termination. After termination this function is not called anymore,
		# thus we can just return 0 elsewhere in the code.
		if self.rewardFunction == None:
			if self.matrixMDP[nextX][nextY] == -1 \
				or self._getStateIndex(nextX, nextY) == self.numStates:
				return 0
			else:
				return -1

		# I first look at the state I am in
		currStateIdx = self._getStateIndex(currX, currY)
		# Now I can look at the next state
		nextStateIdx = self._getStateIndex(nextX, nextY)

		# Now I can finally compute the reward
		reward = self.rewardFunction[nextStateIdx] \
			- self.rewardFunction[currStateIdx]

		return reward

	def reset(self):
		''' Resets the agent to its initial position.'''
		self.currX = self.startX
		self.currY = self.startY

	def isTerminal(self):
		''' Returns whether the agent is in a terminal state (or goal).'''
		if self.currX == self.goalX and self.currY == self.goalY:
			return True
		else:
			return False

	def act(self, action):
		''' At first there are four possible actions: up, down, left and right.
		If the agent tries to go to a -1 state it will stay on the same coord.
		I decided to not implement any stochasticity for now.'''

		# Basically I get what will be the next state and before really making
		# it my current state I verify everything is sound (it is terminal only
		# if we are not using eigenpurposes).
		if self.rewardFunction == None and self.isTerminal():
			return 0
		else:
			nextX, nextY = self._getNextState(action)
			reward = self._getNextReward(
				self.currX, self.currY, action, nextX, nextY)
			self.currX = nextX
			self.currY = nextY
			return reward

	def getGridDimensions(self):
		''' Returns gridworld width and height.'''
		return self.numRows, self.numCols

	def getNumStates(self):
		''' Returns the total number of states (including walls) in the MDP.'''
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

	def getNextStateAndReward(self, currState, action):
		''' One step forward model: return the next state and reward given an
		observation. '''

		# In case it is the absorbing state encoding end of an episode
		if currState == self.numStates:
			return currState, 0

		# First I'll save the original state the agent is on
		currStateIdx = self.getCurrentState()
		# Now I can reset the agent to the state I was told to
		tempX = self.currX
		tempY = self.currY
		self.currX, self.currY = self.getStateXY(currState)

		# Now I can ask what will happen next in this new state
		nextStateIdx = None
		reward = None
		if self.rewardFunction == None and self.isTerminal():
			nextStateIdx = self.numStates
			reward = 0
		else:
			nextX, nextY = self._getNextState(action)
			if nextX != -1 and nextY != -1: # If it is not the absorbing state:
				reward = self._getNextReward(
					self.currX, self.currY, action, nextX, nextY)
				nextStateIdx = self._getStateIndex(nextX, nextY)
			else:
				reward = 0
				nextStateIdx = self.numStates

		# We need to restore the previous configuration:
		self.currX = tempX
		self.currY = tempY

		return nextStateIdx, reward

	def getNextStateAndRewardFromOption(self, currState, o_pi, actionSet):
		'''Execute option until it terminates. It will always terminate. We
			then return the number of time steps it took (-reward) and the
			terminal state.'''

		# In case it is the absorbing state encoding end of an episode
		if currState == self.numStates:
			return currState, 0

		# First I'll save the original state the agent is on
		currStateIdx = self.getCurrentState()
		goalIdx = self._getStateIndex(self.goalX, self.goalY)
		# Now I can reset the agent to the state I was told to
		tempX = self.currX
		tempY = self.currY

		self.currX, self.currY = self.getStateXY(currState)

		# Now I can ask what will happen next in this new state
		accum_reward = 0
		nextStateIdx = currState

		aTerminate = len(actionSet) - 1
		nextAction = o_pi[currState]

		# I need these contour cases for the termination:
		if currState == goalIdx:
			nextStateIdx = self.numStates
		elif nextAction == aTerminate and \
			self.matrixMDP[self.currX][self.currY] != -1:
			accum_reward = -1

		while nextAction != aTerminate:
			nextAction = o_pi[currState]
			self.currX, self.currY = self.getStateXY(currState)
			if self.rewardFunction == None and self.isTerminal():
				nextStateIdx = self.numStates
				nextAction = aTerminate
				reward = 0
			else:
				nextX, nextY = self._getNextState(actionSet[nextAction])
				# If it is not the absorbing state:
				if nextX != -1 and nextY != -1:
					reward = self._getNextReward(
						self.currX, self.currY, nextAction, nextX, nextY)
					nextStateIdx = self._getStateIndex(nextX, nextY)
				else:
					reward = 0
					nextStateIdx = self.numStates

			accum_reward += reward
			currState = nextStateIdx

		# We need to restore the previous configuration:
		self.currX = tempX
		self.currY = tempY

		return nextStateIdx, accum_reward

	def defineRewardFunction(self, vector):
		''' Load vector that will define the reward function: the dot product
		    between the loaded vector and the feature representation.'''
		self.rewardFunction = vector

	def defineGoalState(self, idx):
		''' Returns True if the goal was properly set, otherwise returns False.
		    One may fail to set a goal if it tries to do so in a wall state, in
		    an invalid index, etc.'''

		x, y = self.getStateXY(idx)

		if self.adjMatrix == None:
			self._fillAdjacencyMatrix()

		if idx >= self.numStates:
			return False
		elif self.matrixMDP[x][y] == -1:
			return False
		else:
			self.goalX = x
			self.goalY = y
			self.resetEnvironment()
			return True

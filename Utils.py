'''
This class implements simple methods that are useful in several different places
but they are not more than simple procedures.

Author: Marlos C. Machado
'''
import numpy as np

class Utils:
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

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

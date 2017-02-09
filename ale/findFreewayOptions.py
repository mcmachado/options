import sys
import numpy as np
import matplotlib.pylab as plt

np.set_printoptions(threshold=np.nan)

threshold = 0

# Read arguments
if(len(sys.argv) < 2):
  print("Usage ./findFreewayOptions.py <PATH_TO_OPTIONS>")
  sys.exit()

# Initialize ALE
pathToLoad = sys.argv[1]

x_length1stOption = []
y_length1stOption = []
x_avgLengthOptions = []
y_avgLengthOptions = []
x_stdLengthOptions = []
y_stdLengthOptions = []
x_totalNumUpActions = []
y_totalNumUpActions = []

for i in xrange(1024):
  fileName = pathToLoad + str(i) + '.npy'
  policy = np.load(fileName)

  indicesUpAction = np.where(policy == 1)
  indicesNoOp = np.where(policy == -1)

  #I compute the number of times the action up was pressed
  if len(indicesUpAction[0]) > 0:
    numUpActions = len(indicesUpAction[0])
  else:
    numUpActions = 0

  if numUpActions > threshold:
    x_totalNumUpActions.append(i)
    y_totalNumUpActions.append(numUpActions)

  #I compute the number of times the option chose to act
  if len(indicesNoOp[0]) > 0:
    firstNoOp = indicesNoOp[0][0]
  else:
    firstNoOp = len(policy) - 1
    indicesNoOp = [[]]

  if firstNoOp > threshold:
    x_length1stOption.append(i)
    y_length1stOption.append(firstNoOp)

  #Now I also compute the average length of an option.
  #This is different from the length of the first option.
  lengthOptions = []
  lastIndex = 0
  for j in xrange(len(indicesNoOp[0])):
    lengthOptions.append(indicesNoOp[0][j] - lastIndex - 1)
    lastIndex = indicesNoOp[0][j]

  stdLengthOfOption = np.std(lengthOptions)
  avgLengthOfOption = np.mean(lengthOptions)
  if avgLengthOfOption > threshold:
    x_avgLengthOptions.append(i)
    x_stdLengthOptions.append(i)
    y_avgLengthOptions.append(avgLengthOfOption)
    y_stdLengthOptions.append(stdLengthOfOption)


#for i in xrange(len(x_stdLengthOptions)):
#  print x_stdLengthOptions[i], "%.2f" % y_stdLengthOptions[i]

  if i == 373 or i == 385 or i == 452 or i == 823 or i == 981:
    print i, firstNoOp, avgLengthOfOption, stdLengthOfOption, numUpActions

plt.ylim([0,2000])
plt.scatter(x_length1stOption, y_length1stOption, c='gray', alpha=0.5, label='length 1st option')
plt.scatter(x_totalNumUpActions, y_totalNumUpActions, marker='s', c='red', alpha=0.5, label='total num. ups')
plt.scatter(x_avgLengthOptions, y_avgLengthOptions, marker='^', c='blue', alpha=0.5, label ='avg. option length')
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.xlabel('option #')
plt.show()
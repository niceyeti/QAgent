from __future__ import print_function
import numpy as np
import sys
from sklearn import linear_model

def usage():
	print("usage: python ./linregress.py [csv file] optional: --addBias (add 1's column to data)")

if len(sys.argv) < 2:
	print("wrong number of args: "+str(len(sys.argv)))
	usage()
	exit()

addBiasCol = False
fname = sys.argv[1]
for arg in sys.argv:
	if "--addBias" == arg:
		addBiasCol = True

print("Running linear regression. MAKE SURE STATE DIMENSION IS CORRECT: 3")
data = np.genfromtxt(fname,delimiter=',')
xs = data[:,[0,1,2]]
#some data may have only the x-values (sensor/state values); this adds a column of one's as a bias to give linregress another degree of freedom
if addBiasCol:
	xs = np.append(xs, np.ones([len(xs),1]),1)
	print(str(xs))
	print("new x's")

ys = data[:,[3]]
regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)
print("fitting examples of dim "+str(data.shape)+" from "+fname)
regr.fit(xs, ys)
print("Coefficients: "+str(regr.coef_))
print("Coefficient of determination: "+str(regr.score(xs,ys)))




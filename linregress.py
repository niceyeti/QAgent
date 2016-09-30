from __future__ import print_function
import numpy as np
import sys
from sklearn import linear_model

if len(sys.argv) != 2:
	print("wrong number of args: "+str(len(sys.argv)))
	print("usage: python ./linregress.py [csv file]")
	exit()

fname = sys.argv[1]

print("Running linear regression. MAKE SURE STATE DIMENSION IS CORRECT: 3")
data = np.genfromtxt(fname,delimiter=',')
xs = data[:,[0,1,2]]
ys = data[:,[3]]
regr = linear_model.LinearRegression()
print("fitting examples of dim "+str(data.shape)+" from "+fname)
regr.fit(xs, ys)
print("Coefficients: "+str(regr.coef_))
print("Coefficient of determination: "+str(regr.score(xs,ys)))




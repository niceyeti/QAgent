"""
Just a wrapper for the python logistic regression libs, to which
I'm outsourcing the work of logistic regression.

Input file is formatted as:
	-0.133378,-1.44721,0,-1,t
	-0.140145,-1.83205,0,-1,t
	-0.147399,-1.97014,0,-1,s
	-0.155181,-1.99746,0,-1,c
	where -1th val is alpha (class label), -2th value is the reward value (not used by logistic regression)


The q2agent receives a bunch of labeled-external rewards of the form <x,y,alpha>
where alpha is some char id for the event, x the state vector (inuts), and y the output,
which will be compiled in one big file of such vectors.

So this essentially turns into multiclass regression. Each alpha has a set of logistic regression
parameters, where p(alpha) is taken wrt to each positive instance (vectors for which alpha is
the current alpha being estimated), and all other instances are interpreted as the ~alpha instances.

NOTE this does not implement multinomial regression, since the assumption is that each external
event is independent, such that two different events (crashed, die, etc) could have simultaneously
for the same state vector.

Algorithm:
	Input: A file of labelled reward vectors, as described above.

	-Partition the reward vectors into sets for each alpha label
	foreach alpha:
		-Run logistic regression on each alpha vector set, defining all remaining vectors as ~alpha
		-Write out the parameters as "alpha,p0,p1,p2..."

"""
from __future__ import print_function
import sys
from sklearn import linear_model

def usage():
	print("Usage: python ./logistic.py [input file] [output path]")

if len(sys.argv) < 3:
	print("ERROR too few parameters.")
	usage()
	exit()

ifile = open(sys.argv[1],"r")
ofile = open(sys.argv[2],"w+")

#partition the data into a dict of lists, where lists all have the same alpha label and alpha is key
alphaValues = {} #likely not used, but might need be desirable to preserve in the data for other maximization procs
alphaDict = {}
for line in ifile.readlines():
	#alpha is column 0 of csv
	csvToks = line.strip().split(",")
	alpha = csvToks[0]
	#no item yet, so create on item list
	if alpha not in alphaDict.keys():
		alphaDict[alpha] = []
		#vec is formatted as [x1, x2, x3, y], this stores only the x portion
		vec = csvToks[1:-1]
		alphaDict[alpha].append(vec)
		#store the "value" of this event, which is assumed constant whenever it occurs
		alphaValues[alpha] = csvToks[-1]
	else:
		alphaDict[alpha].append(csvToks[1:-1])

lr = linear_model.LogisticRegression()

#build each dataset, run logistic regression, and output the parameters to file
for alpha in alphaDict.keys():
	#build the datasets to be input to fit()
	posXs = alphaDict[alpha]
	negXs = []
	for other in alphaDict.keys():
		if other != alpha:
			negXs.append(alphaDict[other])
	#create the binary output column: 1 for alpha's vectors, 0 for the others
	Ys = [1 for i in range(0,len(posXs))] + [0 for i in range(0,len(negXs))]
	Xs = posXs + negXs
	lr.fit(Xs,Ys)
	for c in lr.coefs_:
		ofile.write(str(c)+",")
	ofile.write(alpha)

ofile.close()
ifile.close()


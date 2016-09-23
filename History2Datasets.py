"""
Converts the agent's historical record of experiences to datasets suitable for offline
ann training. The agent's experiences are formatted as:

	<-0.707107,10,1.41421>,-7.49795,2.75585,DOWN

Where the <...> is the state vector, the second to last value the target network value, the
last value the estimated value.
Currently there are four actions: UP, DOWN, LEFT, RIGHT. So this script writes out a
dataset for each action: down.txt, up.txt, etc. containing the experiences corresponding
to that action.
"""

import sys

"""
@state: a list of strings representing floats
@qTarget: the target network value as a string
@ofile: the file handle to which this example should be written
"""
def writeRecord(state, qTarget, ofile):
	for val in state:
		ofile.write(val+",")
	ofile.write(qTarget+"\n")


historyFile = open(sys.argv[1],"r")

upFile = open("Data/up.txt","w+")
downFile = open("Data/down.txt","w+")
leftFile = open("Data/left.txt","w+")
rightFile = open("Data/right.txt","w+")

for record in historyFile.readlines():
	stateVals = record.split(">")[0].replace("<","").split(",")
	action = record.strip().split(",")[-1]
	target = record.strip().split(",")[-2]
	estimate = record.strip().split(",")[-3]

	#print(action+","+target+","+estimate)

	if action == "UP":
		writeRecord(stateVals, target, upFile)
	if action == "DOWN":
		writeRecord(stateVals, target, downFile)
	if action == "LEFT":
		writeRecord(stateVals, target, leftFile)
	if action == "RIGHT":
		writeRecord(stateVals, target, rightFile)

upFile.close()
downFile.close()
leftFile.close()
rightFile.close()



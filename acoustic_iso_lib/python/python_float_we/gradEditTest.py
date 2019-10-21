#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Laplacian2d

# Solver library
import pyOperator as pyOp
import wriUtilFloat

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)

	# get params
	gradInFile=parObject.getString("gradIn","noModelFile")
	gradOutFile=parObject.getString("gradOut","noDataFile")

	#read in grad
	gradIn=genericIO.defaultIO.getVector(gradInFile)

	gradEdit = parObject.getInt("gradEdit",0)
	if(gradEdit==1):
		gradEditOp=wriUtilFloat.grad_edit_mora
	elif(gradEdit==2):
		gradEditOp=wriUtilFloat.grad_edit_diving
	else:
		gradEditOp=None

	#edit grad
	gradOut = gradIn.clone()
	gradIn.getNdArray()[:] = gradEditOp(gradIn.getNdArray())

	#write out grad
	genericIO.defaultIO.writeVector(gradOutFile,gradIn)

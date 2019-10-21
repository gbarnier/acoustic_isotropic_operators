#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

import pyOperator as pyOp
import wriUtilFloat
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)

	############################# Initialization ###############################
	# forcing term op
	if(pyinfo): print("--------------------------- forcing term op init --------------------------------")
	forcingTermOp,prior = wriUtilFloat.forcing_term_op_init(sys.argv)

	# # Forward
	# if (parObject.getInt("adj",0) == 0):
	# 	print("-------------------------------------------------------------------")
	# 	print("--------- Running Python regular data extraction forward  -----------")
	# 	print("-------------------------------------------------------------------\n")
	#
	# 	# Read  model
	# 	modelFloat=genericIO.defaultIO.getVector(modelFile)
	#
	# 	dataFloat.scale(0.0)
	#
	# 	################################ DP Test ###################################
	# 	if (parObject.getInt("dp",0)==1):
	# 		print("\nData op dp test:")
	# 		dataSamplingOp.dotTest(1)
	#
	# 	#run forward
	# 	dataSamplingOp.forward(False,modelFloat,dataFloat)
	#
	# 	#write data to disk
	# 	genericIO.defaultIO.writeVector(dataFile,dataFloat)
	#
	#
	# else:
	# 	print("-------------------------------------------------------------------")
	# 	print("--------- Running Python regular data extraction adjoint -----------")
	# 	print("-------------------------------------------------------------------\n")
	#
	# 	# Data
	# 	dataFloat=genericIO.defaultIO.getVector(dataFile)
	#
	# 	modelFloat.scale(0.0)
	#
	# 	################################ DP Test ###################################
	# 	if (parObject.getInt("dp",0)==1):
	# 		print("\nData op dp test:")
	# 		dataSamplingOp.dotTest(1)
	#
	# 	#run adjoint
	# 	dataSamplingOp.adjoint(False,modelFloat,dataFloat)
	#
	# 	#write model to disk
	# 	genericIO.defaultIO.writeVector(modelFile,modelFloat)

	#write prior to disk
	genericIO.defaultIO.writeVector(parObject.getString("priorFile","./priorFile.H"),prior)

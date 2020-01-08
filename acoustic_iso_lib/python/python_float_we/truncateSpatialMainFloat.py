#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import TruncateSpatialReg

# Solver library
import pyOperator as pyOp
import wriUtilFloat
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)

	############################# Initialization ###############################
	# Data extraction
	if(pyinfo): print("--------------------------- Data extraction init --------------------------------")
	modelFloat,dataFloat,dataSamplingOp = wriUtilFloat.data_extraction_reg_op_init(sys.argv)

	# Check that model was provided
	modelFile=parObject.getString("model","noModelFile")
	if (modelFile == "noModelFile"):
		print("**** ERROR: User did not provide model file ****\n")
		quit()
	dataFile=parObject.getString("data","noDataFile")
	if (dataFile == "noDataFile"):
		print("**** ERROR: User did not provide data file name ****\n")
		quit()

	print("*** domain and range checks *** ")
	print("* Kp - d * ")
	print("K domain: ", dataSamplingOp.getDomain().getNdArray().shape)
	print("p shape: ", modelFloat.getNdArray().shape)
	print("K range: ", dataSamplingOp.getRange().getNdArray().shape)
	print("d shape: ", dataFloat.getNdArray().shape)

	# Forward
	if (parObject.getInt("adj",0) == 0):
		print("-------------------------------------------------------------------")
		print("--------- Running Python regular data extraction forward  -----------")
		print("-------------------------------------------------------------------\n")

		# Read  model
		modelFloat=genericIO.defaultIO.getVector(modelFile)

		dataFloat.scale(0.0)

		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			dataSamplingOp.dotTest(1)

		#run forward
		dataSamplingOp.forward(False,modelFloat,dataFloat)

		#write data to disk
		genericIO.defaultIO.writeVector(dataFile,dataFloat)


	else:
		print("-------------------------------------------------------------------")
		print("--------- Running Python regular data extraction adjoint -----------")
		print("-------------------------------------------------------------------\n")

		# Data
		dataFloat=genericIO.defaultIO.getVector(dataFile)

		modelFloat.scale(0.0)

		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			dataSamplingOp.dotTest(1)

		#run adjoint
		dataSamplingOp.adjoint(False,modelFloat,dataFloat)

		#write model to disk
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

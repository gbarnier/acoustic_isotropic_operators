#!/usr/bin/env python3.6
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import SampleWfld

# Solver library
import pyOperator as pyOp
import wriUtilFloat
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# io stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	pyinfo=parObject.getInt("pyinfo",1)


	# get params 
	modelFile=parObject.getString("model","noModelFile")
	dataFile=parObject.getString("data","noDataFile")

	# Forward
	if (parObject.getInt("adj",0) == 0):
		print("-------------------------------------------------------------------")
		print("--------- Running Python regular data extraction forward  -----------")
		print("-------------------------------------------------------------------\n")

		# Read  model
		modelFloat=genericIO.defaultIO.getVector(modelFile)

		############################# Initialization ###############################
		# SampleWfld init
		if(pyinfo): print("--------------------------- SampleWfld init --------------------------------")
		_,dataFloat,sampleWfldOp= wriUtilFloat.wfld_extraction_reg_op_init(sys.argv)
		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", sampleWfldOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", sampleWfldOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			sampleWfldOp.dotTest(1)

		#run forward
		sampleWfldOp.forward(False,modelFloat,dataFloat)

		#write data to disk
		genericIO.defaultIO.writeVector(dataFile,dataFloat)


	else:
		print("-------------------------------------------------------------------")
		print("--------- Running Python regular data extraction adjoint -----------")
		print("-------------------------------------------------------------------\n")

		# Data
		dataFloat=genericIO.defaultIO.getVector(dataFile)

		############################# Initialization ###############################
		# SampleWfld init
		if(pyinfo): print("--------------------------- SampleWfld init --------------------------------")
		modelFloat,_,sampleWfldOp= wriUtilFloat.wfld_extraction_reg_op_init(sys.argv)
		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", sampleWfldOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", sampleWfldOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			sampleWfldOp.dotTest(1)

		#run adjoint
		sampleWfldOp.adjoint(False,modelFloat,dataFloat)

		#write model to disk
		genericIO.defaultIO.writeVector(modelFile,modelFloat)



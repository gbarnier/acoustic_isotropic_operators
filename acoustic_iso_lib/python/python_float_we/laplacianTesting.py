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
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
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

		dataFloat=modelFloat.clone()
		dataFloat.scale(0.0)

		############################# Initialization ###############################
		# SampleWfld init
		if(pyinfo): print("--------------------------- SampleWfld init --------------------------------")
		laplOp= Laplacian2d.laplacian2d(modelFloat,dataFloat)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", laplOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", laplOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			laplOp.dotTest(1)

		#run forward
		laplOp.forward(False,modelFloat,dataFloat)

		#write data to disk
		genericIO.defaultIO.writeVector(dataFile,dataFloat)


	else:
		print("-------------------------------------------------------------------")
		print("--------- Running Python regular data extraction adjoint -----------")
		print("-------------------------------------------------------------------\n")

		# Data
		dataFloat=genericIO.defaultIO.getVector(dataFile)

		modelFloat=dataFloat.clone()
		modelFloat.scale(0.0)

		############################# Initialization ###############################
		# SampleWfld init
		if(pyinfo): print("--------------------------- SampleWfld init --------------------------------")
		laplOp= Laplacian2d.laplacian2d(modelFloat,dataFloat)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", laplOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", laplOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			laplOp.dotTest(1)

		#run adjoint
		laplOp.adjoint(False,modelFloat,dataFloat)

		#write model to disk
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

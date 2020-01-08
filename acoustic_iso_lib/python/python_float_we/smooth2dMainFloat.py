#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Smooth2d

# Solver library
import pyOperator as pyOp
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)


	# get params
	modelFile=parObject.getString("model","noModelFile")
	dataFile=parObject.getString("data","noDataFile")
	nfilt1=parObject.getInt("nfilt1",-1)
	nfilt2=parObject.getInt("nfilt2",-1)
	if(nfilt1==-1 or nfilt2==-1 ):
		print("**** ERROR: User did not provide one of the nfilt1/nfilt2 sizes ****\n")
		quit()



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
		smooth2dOp = Smooth2d.smooth2d(modelFloat,dataFloat,nfilt1,nfilt2)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", smooth2dOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", smooth2dOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			smooth2dOp.dotTest(1)

		#run forward
		smooth2dOp.forward(False,modelFloat,dataFloat)

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
		smooth2dOp = Smooth2d.smooth2d(modelFloat,dataFloat,nfilt1,nfilt2)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", smooth2dOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", smooth2dOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			smooth2dOp.dotTest(1)

		#run adjoint
		smooth2dOp.adjoint(False,modelFloat,dataFloat)

		#write model to disk
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

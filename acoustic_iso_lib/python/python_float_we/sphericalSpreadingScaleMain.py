#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import SphericalSpreadingScale

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

		############################# Initialization ###############################
		# SphericalSpreadingScale init
		if(pyinfo): print("--------------------------- SphericalSpreadingScale init --------------------------------")
		pressureData=parObject.getString("pressureData","noPressureData")
		if(pressureData == "noPressureData"):
			_,dataFloat,shericalSpreadingOp= wriUtilFloat.spherical_spreading_op_init(sys.argv)
		else:
			pressure=genericIO.defaultIO.getVector(pressureData)
			shericalSpreadingOp= SphericalSpreadingScale.spherical_spreading_scale_wfld(modelFloat,modelFloat,pressure)
			dataFloat=modelFloat.clone()
		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", shericalSpreadingOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", shericalSpreadingOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			shericalSpreadingOp.dotTest(1)

		#run forward
		shericalSpreadingOp.forward(False,modelFloat,dataFloat)

		#write data to disk
		genericIO.defaultIO.writeVector(dataFile,dataFloat)


	else:
		print("-------------------------------------------------------------------")
		print("--------- Running Python regular data extraction adjoint -----------")
		print("-------------------------------------------------------------------\n")

		# Data
		dataFloat=genericIO.defaultIO.getVector(dataFile)

		############################# Initialization ###############################
		# SphericalSpreadingScale init
		if(pyinfo): print("--------------------------- SphericalSpreadingScale init --------------------------------")
		pressureData=parObject.getString("pressureData","noPressureData")
		if(pressureData == "noPressureData"):
			modelFloat,_,shericalSpreadingOp= wriUtilFloat.spherical_spreading_op_init(sys.argv)
		else:
			pressure=genericIO.defaultIO.getVector(pressureData)
			shericalSpreadingOp= SphericalSpreadingScale.spherical_spreading_scale_wfld(dataFloat,dataFloat,pressure)
			modelFloat=dataFloat.clone()
		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", shericalSpreadingOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", shericalSpreadingOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			shericalSpreadingOp.dotTest(1)

		#run adjoint
		shericalSpreadingOp.adjoint(False,modelFloat,dataFloat)

		#write model to disk
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

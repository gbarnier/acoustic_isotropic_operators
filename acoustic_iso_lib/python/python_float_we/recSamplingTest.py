#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os
import math

# Modeling operators
import Acoustic_iso_float_we
import wriUtilFloat

if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------ receiver sampling test ----------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")


	# get params
	modelFile=parObject.getString("model","noModelFile")
	dataFile=parObject.getString("data","noDataFile")

	# Forward
	if (parObject.getInt("adj",0) == 0):
		print("-------------------------------------------------------------------")
		print("--------- Running Python regular data extraction forward  -----------")
		print("-------------------------------------------------------------------\n")


                ############################# Initialization ##############################
		# Data extraction
		if(pyinfo): print("--------------------------- Data extraction init --------------------------------")
		dataSamplingOp,modelFloat,dataFloat = wriUtilFloat.data_extraction_op_init(sys.argv)

		# Read  model
		modelFloat=genericIO.defaultIO.getVector(modelFile)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", dataSamplingOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", dataSamplingOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
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
		print("--------- Running Python recevier extraction adjoint -----------")
		print("-------------------------------------------------------------------\n")


		############################# Initialization ###############################
		# SampleWfld init
		if(pyinfo): print("--------------------------- SampleWfld init --------------------------------")
		dataSamplingOp,modelFloat,dataFloat = wriUtilFloat.data_extraction_op_init(sys.argv)

		# Data
		dataFloat=genericIO.defaultIO.getVector(dataFile)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", dataSamplingOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", dataSamplingOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			dataSamplingOp.dotTest(1)

		#run adjoint
		dataSamplingOp.adjoint(False,modelFloat,dataFloat)

		#write model to disk
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

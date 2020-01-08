#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import SecondDeriv

# Solver library
import pyOperator as pyOp
from sys_util import logger

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
		print("--------- Running Python second time deriv forward  -----------")
		print("-------------------------------------------------------------------\n")

		# Read  model
		modelFloat=genericIO.defaultIO.getVector(modelFile)

		dataFloat=modelFloat.clone()
		dataFloat.scale(0.0)

		############################# Initialization ###############################
		# TpowWfld init
		if(pyinfo): print("--------------------------- second time deriv init --------------------------------")
		secondDerivMultExpOp = SecondDeriv.second_deriv_multi_exp(modelFloat,dataFloat)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", secondDerivMultExpOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", secondDerivMultExpOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			secondDerivMultExpOp.dotTest(1)

		#run forward
		secondDerivMultExpOp.forward(False,modelFloat,dataFloat)

		#write data to disk
		genericIO.defaultIO.writeVector(dataFile,dataFloat)


	else:
		print("-------------------------------------------------------------------")
		print("--------- Running Python second time deriv adjoint -----------")
		print("-------------------------------------------------------------------\n")

		# Data
		dataFloat=genericIO.defaultIO.getVector(dataFile)

		modelFloat=dataFloat.clone()
		modelFloat.scale(0.0)

		############################# Initialization ###############################
		# TpowWfld init
		if(pyinfo): print("--------------------------- second time deriv init --------------------------------")
		secondDerivMultExpOp = SecondDeriv.second_deriv_multi_exp(modelFloat,dataFloat)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", secondDerivMultExpOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", secondDerivMultExpOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			secondDerivMultExpOp.dotTest(1)

		#run adjoint
		secondDerivMultExpOp.adjoint(False,modelFloat,dataFloat)

		#write model to disk
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

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

	# Initialize operator
	modelFloat,dataFloat,parObject,secondDerivMultExpOp,fftOp,timeFloat = SecondDeriv.secondDerivOpInitFloat_multi_exp_freq(sys.argv)

	print("*** domain and range checks *** ")
	print("* Kp - d * ")
	print("K domain: ", secondDerivMultExpOp.getDomain().getNdArray().shape)
	print("p shape: ", modelFloat.getNdArray().shape)
	print("K range: ", secondDerivMultExpOp.getRange().getNdArray().shape)
	print("d shape: ", dataFloat.getNdArray().shape)

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

		inputMode=parObject.getString("inputMode","freq")
		if (inputMode == 'time'):
			print('------ input model in time domain. converting to freq ------')
			timeFloat = genericIO.defaultIO.getVector(modelFile)
			fftOp.adjoint(0,modelFloat,timeFloat)
			genericIO.defaultIO.writeVector('freq_model.H',modelFloat)
		else:
			modelFloat=genericIO.defaultIO.getVector(modelFile)

		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			secondDerivMultExpOp.dotTest(1)

		#run forward
		secondDerivMultExpOp.forward(False,modelFloat,dataFloat)

		#check if provided in time domain
		outputMode=parObject.getString("outputMode","freq")
		if (outputMode == 'time'):
			print('------ output mode is time domain. converting to time ------')
			fftOp.forward(0,dataFloat,timeFloat)
			#write data to disk
			genericIO.defaultIO.writeVector(dataFile,timeFloat)
			genericIO.defaultIO.writeVector('freq_data.H',dataFloat)
		else:
			genericIO.defaultIO.writeVector(dataFile,dataFloat)


	else:
		print("-------------------------------------------------------------------")
		print("--------- Running Python second time deriv adjoint -----------")
		print("-------------------------------------------------------------------\n")

		#check if provided in time domain
		inputMode=parObject.getString("inputMode","freq")
		if (inputMode == 'time'):
			print('------ input data in time domain. converting to freq ------')
			timeFloat = genericIO.defaultIO.getVector(modelFile)
			fftOp.adjoint(0,dataFloat,timeFloat)
		else:
			modelFloat=genericIO.defaultIO.getVector(modelFile)

		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			secondDerivMultExpOp.dotTest(1)

		#run adjoint
		secondDerivMultExpOp.adjoint(False,modelFloat,dataFloat)

		#check if provided in time domain
		outputMode=parObject.getString("outputMode","freq")
		if (outputMode == 'time'):
			print('------ output mode is time domain. converting to time ------')
			fftOp.forward(0,modelFloat,timeFloat)
			#write data to disk
			genericIO.defaultIO.writeVector(modelFile,timeFloat)
		else:
			genericIO.defaultIO.writeVector(modelFile,modelFloat)

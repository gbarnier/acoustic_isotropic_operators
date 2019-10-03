#!/usr/bin/env python3.6
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Mask3d 

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
	vel_max=parObject.getFloat("vel_max",-1)
	t_min=parObject.getFloat("t_min",-1)
	source_ix=parObject.getInt("ix",-1)
	source_iz=parObject.getInt("iz",-1)
	if(vel_max==-1 or t_min==-1 or source_ix==-1 or source_iz==-1):
		print("**** ERROR: User did not provide one of the min/max parameters ****\n")
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
		# CausalMask init
		if(pyinfo): print("--------------------------- CausalMask init --------------------------------")
		causalMaskOp = Mask3d.causalMask(modelFloat,dataFloat,vel_max,t_min,source_ix,source_iz)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", causalMaskOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", causalMaskOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			causalMaskOp.dotTest(1)

		#run forward
		causalMaskOp.forward(False,modelFloat,dataFloat)

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
		# CausalMask init
		if(pyinfo): print("--------------------------- CausalMask init --------------------------------")
		causalMaskOp = Mask3d.causalMask(modelFloat,dataFloat,vel_max,t_min,source_ix,source_iz)

		print("*** domain and range checks *** ")
		print("* Kp - d * ")
		print("K domain: ", causalMaskOp.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("K range: ", causalMaskOp.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			causalMaskOp.dotTest(1)

		#run adjoint
		causalMaskOp.adjoint(False,modelFloat,dataFloat)

		#write model to disk
		genericIO.defaultIO.writeVector(modelFile,modelFloat)



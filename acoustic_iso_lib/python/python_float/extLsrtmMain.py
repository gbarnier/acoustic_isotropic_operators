#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import dataTaperModule
import dsoGpuModule
import numpy as np
import time
import sys

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger

# Template for linearized waveform inversion 
if __name__ == '__main__':

	# I/O Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	dataTaper=parObject.getInt("dataTaper",0)
	reg=parObject.getInt("reg",0)
	regType=parObject.getString("regType","dso")

	############################################################################
	############################# Inversion operators ##########################
	############################################################################

	######## Conventional ########
	if (dataTaper==0):

		print("-------------------------------------------------------------------")
		print("------------ Conventional extended linearized inversion -----------")
		print("-------------------------------------------------------------------\n")

		# Born extended initialization
		modelStart,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornExtOpInitFloat(sys.argv)

		# Read starting model
		modelStartFile=parObject.getString("modelStart")
		modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=2)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Born instanciation
		BornExtOp=Acoustic_iso_float.BornExtShotsGpu(modelStart,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)
		invOp=BornExtOp

	######## Data tapering ########
	else:

		print("-------------------------------------------------------------------")
		print("---------- Extended linearized inversion + data tapering ----------")
		print("-------------------------------------------------------------------\n")

		# Born extended initialization
		modelStart,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornExtOpInitFloat(sys.argv)

		# Data taper initialization
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset=dataTaperModule.dataTaperInit(sys.argv)

		# Read starting model
		modelStartFile=parObject.getString("modelStart")
		modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=2)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Born instanciation
		BornExtOp=Acoustic_iso_float.BornExtShotsGpu(modelStart,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

		# Data taper instanciation
		dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset)
		dataTapered=dataTaper.clone()
		dataTaperOp.forward(False,data,dataTapered)
		data=dataTapered

		# Operator chain
		invOp=pyOp.ChainOperator(BornExtOp,dataTaperOp)

	############################################################################
	################################# Problem ##################################
	############################################################################
	# Regularization
	if (reg==1):

		# DSO regularization
		if (regType=="dso"):
			print("DSO regularization")
			nz,nx,nExt,fat,zeroShift=dsoGpuModule.dsoGpuInit(sys.argv)
			dsoOp=dsoGpuModule.dsoGpu(modelStart,data,nz,nx,nExt,fat,zeroShift)
			invProb=Prblm.ProblemL2LinearReg(modelStart,data,invOp,epsilon,reg_op=dsoOp)

		# Identity regularization
		if (regType=="id"):
			print("Identity regularization")

		# Evaluate Epsilon
		if (epsilonEval==1):
			print("Epsilon evaluation")
			invProb.estimate_epsilon()

	# No regularization
	else:
		print("No regularization")
		invProb=Prblm.ProblemL2Linear(modelStart,data,invOp)

	############################################################################
	################################# Solver ###################################
	############################################################################
	# Stopper
	stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

	# Solver
	logFile=parObject.getString("logFile")
	invPrefix=parObject.getString("prefix")
	LCGsolver=LCG.LCGsolver(stop,logger=logger(logFile))
	LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

	# Run solver
	LCGsolver.run(invProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

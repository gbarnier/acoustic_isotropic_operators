#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import dataTaperModule
import numpy as np
import time
import sys

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# I/O Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	dataTaper=parObject.getInt("dataTaper",0)
	reg=parObject.getInt("reg",0)
	# regType=parObject.getString("regType","dso")

	######## Case #1: Conventional extended linearized Born inversion ##########
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

		# Regularization
		if (reg==1):
			if (regType=="dso"):
				print("Not ready yet")
				# # Problem
				# lsrtmProb=Prblm.ProblemL2LinearReg(modelStart,data,BornExtOp,epsilon,reg_op=dsoOp)
				# lsrtmProbReg.estimate_epsilon()

		# No regularization
		else:
			# Problem
			lsrtmProb=Prblm.ProblemL2Linear(modelStart,data,BornExtOp)

	############################################################################

	###### Case #2: Extended linearized Born inversion with data tapering ######
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
		lsrtmOp=pyOp.ChainOperator(BornExtOp,dataTaperOp)

		# Regularization
		if (reg==1):
			if (regType=="dso"):
				print("Not ready yet")
			   #  # Problem
			   # lsrtmProb=Prblm.ProblemL2LinearReg(modelStart,data,lsrtmOp,epsilon,reg_op=dsoOp)
			   # lsrtmProbReg.estimate_epsilon()

		# No regularization
		else:
			# Problem
			lsrtmProb=Prblm.ProblemL2Linear(modelStart,data,lsrtmOp)


	############################################################################

	################################# Solver ###################################
	# Stopper
	stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

	# Solver
	logFile=parObject.getString("logFile")
	invPrefix=parObject.getString("prefix")
	LCGsolver=LCG.LCGsolver(stop,logger=logger(logFile))
	LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

	# Run solver
	LCGsolver.run(lsrtmProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

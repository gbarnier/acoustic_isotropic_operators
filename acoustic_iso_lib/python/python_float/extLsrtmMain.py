#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Acoustic_iso_float
import interpBSpline2dModule
import dataTaperModule
import spatialDerivModule
import dsoGpuModule

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	spline=parObject.getInt("spline",0)
	dataTaper=parObject.getInt("dataTaper",0)
	regType=parObject.getString("reg","None")
	reg=0
	if (regType != "None"): reg=1
	epsilonEval=parObject.getInt("epsilonEval",0)

	print("-------------------------------------------------------------------")
	print("-------------------- Extended linearized inversion ----------------")
	print("-------------------------------------------------------------------\n")

	############################# Initialization ###############################
	# Data taper
	if (dataTaper==1):
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset=dataTaperModule.dataTaperInit(sys.argv)

	# Born
	modelFineInit,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornExtOpInitFloat(sys.argv)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInit=modelFineInit.clone()
		modelInit.scale(0.0)

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	############################# Instanciation ################################
	# Born
	BornExtOp=Acoustic_iso_float.BornExtShotsGpu(modelFineInit,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)
	invOp=BornExtOp

	# Data taper
	if (dataTaper==1):
		print("--- Using data tapering ---")
		dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset)
		dataTapered=data.clone()
		dataTaperOp.forward(False,data,dataTapered) # Apply tapering to the data
		data=dataTapered
		invOp=pyOp.ChainOperator(BornExtOp,dataTaperOp)

	############################# Regularization ###############################
	# Regularization
	if (reg==1):
		# Get epsilon value from user
		epsilon=parObject.getFloat("epsilon",-1.0)

		# Identity regularization
		if (regType=="id"):
			print("--- Identity regularization ---")
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon)

		# Dso
		elif (regType=="dso"):
			print("--- DSO regularization ---")
			nz,nx,nExt,fat,zeroShift=dsoGpuModule.dsoGpuInit(sys.argv)
			dsoOp=dsoGpuModule.dsoGpu(modelInit,modelInit,nz,nx,nExt,fat,zeroShift)
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,reg_op=dsoOp)

		else:
			print("--- Regularization that you have required is not supported by our code ---")
			quit()

		# Evaluate Epsilon
		if (epsilonEval==1):
			print("--- Epsilon evaluation ---")
			epsilonOut=invProb.estimate_epsilon()
			print("--- Epsilon value: ",epsilonOut," ---")
			quit()

	# No regularization
	else:
		invProb=Prblm.ProblemL2Linear(modelInit,data,invOp)

	############################## Solver ######################################
	# Stopper
	stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

	# Folder
	folder=parObject.getString("folder")
	if (os.path.isdir(folder)==False): os.mkdir(folder)
	prefix=parObject.getString("prefix","None")
	if (prefix=="None"): prefix=folder
	invPrefix=folder+"/"+prefix
	logFile=invPrefix+"_logFile"

	# Solver recording parameters
	iterSampling=parObject.getInt("iterSampling",1)
	bufferSize=parObject.getInt("bufferSize",-1)
	if (bufferSize<0): bufferSize=None

	# Solver
	LCGsolver=LCG.LCGsolver(stop,logger=logger(logFile))
	LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling)

	# Run solver
	LCGsolver.run(invProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

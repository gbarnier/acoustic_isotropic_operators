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
import interpBSplineModule
import dataTaperModule
import spatialDerivModule
import dsoGpuModule
import dsoInvGpuModule

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pySymLCGsolver as SymLCGsolver
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	pyinfo=parObject.getInt("pyinfo",1)

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------- Preconditioned Extended linearized inversion --------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("-------------------- Extended linearized inversion ----------------")

	############################# Initialization ###############################
	# Born extended
	modelFineInit,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornExtOpInitFloat(sys.argv)

	# Used by both (Wd + Born extended) and Wm (by means of extended modeling)
	seismicData,model,vel,parObject,sourcesVectorDipole,sourcesSignalsVector,receiversVectorDipole,dts,fat,taperEndTraceWidth=Acoustic_iso_float.SymesPseudoInvInit(sys.argv)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInit=modelFineInit.clone()
		modelInit.scale(0.0)
	else:
		modelInit=genericIO.defaultIO.getVector(modelInitFile,ndims=3)

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	############################# Instanciation ################################
	# Born extended Normal
	BornExtOp=Acoustic_iso_float.BornExtShotsGpu(modelFineInit,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)
	invOp=BornExtOp

	BornExtOpAdj=pyOp.Transpose(BornExtOp)


	# Wd + born extended
	wdBornExtendedOp=Acoustic_iso_float.SymesWdBornExtGpu(seismicData,model,vel,parObject,sourcesVectorDipole,sourcesSignalsVector,receiversVectorDipole,dts,fat,taperEndTraceWidth)

	# Wm
	wmOp=Acoustic_iso_float.SymesWmGpu(model,model,vel,fat)

	wdOp=Acoustic_iso_float.SymesWdGpu(seismicData,dts)

	# wdOp.powerMethod(True,n_iter=100,eval_min=True)



	# Concatenate operators
	invOp=pyOp.ChainOperator(BornExtOp,wdBornExtendedOp)
	# invOp1=pyOp.ChainOperator(BornExtOp,wdOp)
	# invOp=pyOp.ChainOperator(invOp1,BornExtOpAdj)
	# invOp=pyOp.ChainOperator(BornExtOp,BornExtOpAdj)

	invOp.powerMethod(True,n_iter=10,eval_min=True)
	quit()

	########################## Compute right-hand side #########################
	dataInv=model.clone()

	# BornadjWd=pyOp.ChainOperator(wdOp,BornExtOpAdj)

	wdBornExtendedOp.forward(False,data,dataInv)
	# BornadjWd.forward(False,data,dataInv)

	# Instanciate problem with model preconditioning
	invProb=Prblm.ProblemLinearSymmetric(model,dataInv,invOp)
	# invProb=Prblm.ProblemLinearSymmetric(model,dataInv,invOp)

	############################## Solver ######################################
	# Solver
	symSolver=SymLCGsolver.SymLCGsolver(stop,logger=inv_log)
	symSolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	symSolver.run(invProb,verbose=True)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

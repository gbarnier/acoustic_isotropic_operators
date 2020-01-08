#!/usr/bin/env python3
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

	# IO object
	parObject=genericIO.io(params=sys.argv)

	pyinfo=parObject.getInt("pyinfo",1)
	dataTaper=parObject.getInt("dataTaper",0)

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------- Preconditioned offset-extended linearized inversion -----")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("--------- Preconditioned offset-extended linearized inversion -----")

	############################# Initialization ###############################
	# Data taper
	if (dataTaper==1):
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth=dataTaperModule.dataTaperInit(sys.argv)

	# Born extended
	model,seismicData,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornExtOpInitFloat(sys.argv)

	# Symes' pseudo inverse
	# We use dipole sources/receivers and we apply a "end of trace" tapering to the integrated data before Born extended adjoint
	seismicData,model,vel,parObject,sourcesVectorDipole,sourcesSignalsVector,receiversVectorDipole,dts,fat,taperEndTraceWidth=Acoustic_iso_float.SymesPseudoInvInit(sys.argv)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInit=model.clone()
		modelInit.scale(0.0)
	else:
		modelInit=genericIO.defaultIO.getVector(modelInitFile,ndims=3)

	# Seismic data
	seismicDataFile=parObject.getString("data")
	seismicData=genericIO.defaultIO.getVector(seismicDataFile,ndims=3)

	# Data
	data=model.clone()

	############################# Instanciation ################################
	# Born extended
	BornExtOp=Acoustic_iso_float.BornExtShotsGpu(model,seismicData,vel,parObject.param,sourcesVector,sourcesSignalsVector,receiversVector)

	# Construct Born operator object
	SymesPseudoInvOp=Acoustic_iso_float.SymesPseudoInvGpu(seismicData,model,vel,parObject,sourcesVectorDipole,sourcesSignalsVector,receiversVectorDipole,dts,fat,taperEndTraceWidth)

	# Data taper
	if (dataTaper==1):
		if(pyinfo): print("--- Using data tapering ---")
		inv_log.addToLog("--- Using data tapering ---")
		dataTaperOp=dataTaperModule.datTaper(seismicData,seismicData,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,seismicData.getHyper(),time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth)
		# Apply tapering to the data
		seismicDataTapered=seismicData.clone()
		dataTaperOp.forward(False,seismicData,seismicDataTapered)
		seismicData=seismicDataTapered

	# Apply forward of Pseudo inverse to tapered data
	# Data is the right-side of symmetric system (data = B_dagger seismicData)
	SymesPseudoInvOp.forward(False,seismicData,data)

	# Concatenate operators
	if (dataTaper==1):
		invOpTemp=pyOp.ChainOperator(BornExtOp,dataTaperOp)
		invOp=pyOp.ChainOperator(invOpTemp,SymesPseudoInvOp)
	else:
		invOp=pyOp.ChainOperator(BornExtOp,SymesPseudoInvOp)

	# Create problem
	invProb=Prblm.ProblemLinearSymmetric(model,data,invOp)

	# Run power method
	invOp.powerMethod(True,n_iter=100,eval_min=True,square=True)

	############################## Solver ######################################
	# Solver
	# symSolver=SymLCGsolver.SymLCGsolver(stop,logger=inv_log)
	# symSolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)
	#
	# # Run solver
	# symSolver.run(invProb,verbose=True)

	# Write data
	# dataOutFile=parObject.getString("dataOut")
	# genericIO.defaultIO.writeVector(dataOutFile,data)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

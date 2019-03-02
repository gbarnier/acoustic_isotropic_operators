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
	print("------------------ Conventional linearized inversion --------------")
	print("-------------------------------------------------------------------\n")

	############################# Initialization ###############################
	# Spline
	if (spline==1):
		modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dModule.bSpline2dInit(sys.argv)

	# Data taper
	if (dataTaper==1):
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset=dataTaperModule.dataTaperInit(sys.argv)

	# Born
	modelFineInit,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornOpInitFloat(sys.argv)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (spline==1):
		if (modelInitFile=="None"):
			modelInit=modelCoarseInit.clone()
			modelInit.scale(0.0)
		else:
			modelInit=genericIO.defaultIO.getVector(modelInitFile,ndims=2)
	else:
		if (modelInitFile=="None"):
			modelInit=modelFineInit.clone()
			modelInit.scale(0.0)

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	############################# Instanciation ################################
	# Born
	BornOp=Acoustic_iso_float.BornShotsGpu(modelFineInit,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)
	invOp=BornOp

	# Spline
	if (spline==1):
		print("--- Using spline interpolation ---")
		splineOp=interpBSpline2dModule.bSpline2d(modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)

	# Data taper
	if (dataTaper==1):
		print("--- Using data tapering ---")
		dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset)
		dataTapered=data.clone()
		dataTaperOp.forward(False,data,dataTapered) # Apply tapering to the data
		data=dataTapered

	# Concatenate operators
	if (spline==1 and dataTaper==0):
		invOp=pyOp.ChainOperator(splineOp,BornOp)
	if (spline==0 and dataTaper==1):
		invOp=pyOp.ChainOperator(BornOp,dataTaperOp)
	if (spline==1 and dataTaper==1):
		invOpTemp=pyOp.ChainOperator(splineOp,BornOp)
		invOp=pyOp.ChainOperator(invOpTemp,dataTaperOp)

	############################# Regularization ###############################
	# Regularization
	if (reg==1):
		# Get epsilon value from user
		epsilon=parObject.getFloat("epsilon",-1.0)

		# Identity regularization
		if (regType=="id"):
			print("--- Identity regularization ---")
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon)

		# Spatial gradient in z-direction
		if (regType=="zGrad"):
			print("--- Vertical gradient regularization ---")
			fat=spatialDerivModule.zGradInit(sys.argv)
			gradOp=spatialDerivModule.zGradPython(modelInit,modelInit,fat)
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon)

		# Spatial gradient in x-direction
		if (regType=="xGrad"):
			print("--- Horizontal gradient regularization ---")
			fat=spatialDerivModule.xGradInit(sys.argv)
			gradOp=spatialDerivModule.xGradPython(modelInit,modelInit,fat)
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,reg_op=gradOp)

		# Sum of spatial gradients in z and x-directions
		if (regType=="zxGrad"):
			print("--- Gradient regularization in both directions ---")
			fat=spatialDerivModule.zxGradInit(sys.argv)
			gradOp=spatialDerivModule.zxGradPython(modelInit,modelInit,fat)
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,reg_op=gradOp)

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

	# Solver
	LCGsolver=LCG.LCGsolver(stop,logger=logger(logFile))
	LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

	# Run solver
	LCGsolver.run(invProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

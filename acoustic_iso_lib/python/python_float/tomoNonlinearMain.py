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
import maskGradientModule

# Solver library
import pyOperator as pyOp
from pyNonLinearSolver import NLCGsolver as NLCG
import pyProblem as Prblm
from pyStopper import StopperBase as Stopper
import inversionUtils
from sys_util import logger

# Template for FWI workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	pyinfo=parObject.getInt("pyinfo",1)
	spline=parObject.getInt("spline",0)
	dataTaper=parObject.getInt("dataTaper",0)
	regType=parObject.getString("reg","None")
	reg=0
	if (regType != "None"): reg=1
	epsilonEval=parObject.getInt("epsilonEval",0)
	gradientMask=parObject.getInt("gradientMask",0)

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)

	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------------ Extended tomo nonlinear inversion ---------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------------ Extended tomo nonlinear inversion -------------------------")

	############################# Initialization ###############################
	# Spline
	if (spline==1):
		modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSplineModule.bSpline2dInit(sys.argv)

	# Data taper
	if (dataTaper==1):
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec=dataTaperModule.dataTaperInit(sys.argv)

	# Born extended for tomo nonlienar inversion (model=velocity)
	modelFineInit,data,reflectivity,parObject,sourcesVector,sourcesSignalsVector,receiversVector,_=Acoustic_iso_float.BornExtTomoInvOpInitFloat(sys.argv)

	# Tomo extended
	modelFineInit,data,wavelet,parObject,sourcesVector,receiversVector,reflectivity,_=Acoustic_iso_float.tomoExtOpInitFloat(sys.argv)

	# Gradient mask
	if (gradientMask==1):
		vel,bufferUp,bufferDown,taperExp,fat = maskGradientModule.maskGradientInit(sys.argv)

	############################# Read files ###################################
	# Seismic source
	waveletFile=parObject.getString("sources")
	wavelet=genericIO.defaultIO.getVector(waveletFile,ndims=3)

	# Coarse-grid model
	if (spline==1):
		modelCoarseInitFile=parObject.getString("modelCoarseInit")
		modelCoarseInit=genericIO.defaultIO.getVector(modelCoarseInitFile,ndims=2)

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	############################# Instanciation ################################
	# Born extended for tomo nonlinear inversion
	BornExtTomoInvOp=Acoustic_iso_float.BornExtTomoInvShotsGpu(modelFineInit,data,reflectivity,wavelet,parObject,sourcesVector,receiversVector)

	# Tomo extended
	tomoExtOp=Acoustic_iso_float.tomoExtShotsGpu(modelFineInit,data,modelFineInit,parObject,sourcesVector,sourcesSignalsVector,receiversVector,reflectivity)
	tomoInvOp=tomoExtOp
	if (gradientMask==1):
		maskGradientOp=maskGradientModule.maskGradient(modelFineInit,modelFineInit,vel,bufferUp,bufferDown,taperExp,fat)
		tomoInvOp=pyOp.ChainOperator(maskGradientOp,tomoExtOp)

	# Nonlinear operator (Born + tomo)
	BornExtTomoInvOp=pyOp.NonLinearOperator(BornExtTomoInvOp,tomoInvOp,tomoExtOp.setVel)
	tomoInvOp=tomoNlOp
	modelInit=modelFineInit

	# Spline
	if (spline==1):
		if(pyinfo): print("--- Using spline interpolation ---")
		inv_log.addToLog("--- Using spline interpolation ---")
		modelInit=modelCoarseInit
		splineOp=interpBSplineModule.bSpline2d(modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp) # Create spline nonlinear operator
		BornExtOp.add_spline(splineOp)

	# Data taper
	if (dataTaper==1):
		if(pyinfo): print("--- Using data tapering ---")
		inv_log.addToLog("--- Using data tapering ---")
		dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec)
		dataTapered=data.clone()
		dataTaperOp.forward(False,data,dataTapered) # Apply tapering to the data
		data=dataTapered
		dataTaperNlOp=pyOp.NonLinearOperator(dataTaperOp,dataTaperOp) # Create dataTaper nonlinear operator

	# Concatenate operators
	if (spline==1 and dataTaper==0):
		fwiInvOp=pyOp.CombNonlinearOp(splineNlOp,fwiOp)
	if (spline==0 and dataTaper==1):
		fwiInvOp=pyOp.CombNonlinearOp(fwiOp,dataTaperNlOp)
	if (spline==1 and dataTaper==1):
		fwiInvOpTemp=pyOp.CombNonlinearOp(splineNlOp,fwiOp) # Combine spline and FWI
		fwiInvOp=pyOp.CombNonlinearOp(fwiInvOpTemp,dataTaperNlOp) # Combine everything

	############################# Gradient mask ################################
	maskGradientFile=parObject.getString("maskGradient","NoMaskGradientFile")
	if (maskGradientFile=="NoMaskGradientFile"):
		maskGradient=None
	else:
		maskGradient=genericIO.defaultIO.getVector(maskGradientFile,ndims=2)

	############################### Bounds #####################################
	minBoundVector,maxBoundVector=Acoustic_iso_float.createBoundVectors(parObject,modelInit)

	############################# Regularization ###############################
	# Regularization
	if (reg==1):
		# Get epsilon value from user
		epsilon=parObject.getFloat("epsilon",-1.0)
		inv_log.addToLog("--- Epsilon value: %s ---"%(epsilon))

		# Identity regularization
		if (regType=="id"):
			if(pyinfo): print("--- Identity regularization ---")
			inv_log.addToLog("--- Identity regularization ---")
			fwiProb=Prblm.ProblemL2NonLinearReg(modelInit,data,fwiInvOp,epsilon,grad_mask=maskGradient,minBound=minBoundVector,maxBound=maxBoundVector)

		# Spatial gradient in z-direction
		if (regType=="zGrad"):
			if(pyinfo): print("--- Vertical gradient regularization ---")
			inv_log.addToLog("--- Vertical gradient regularization ---")
			fat=spatialDerivModule.zGradInit(sys.argv)
			gradOp=spatialDerivModule.zGradPython(modelInit,modelInit,fat)
			gradNlOp=pyOp.NonLinearOperator(gradOp,gradOp)
			fwiProb=Prblm.ProblemL2NonLinearReg(modelInit,data,fwiInvOp,epsilon,grad_mask=maskGradient,reg_op=gradNlOp,minBound=minBoundVector,maxBound=maxBoundVector)

		# Spatial gradient in x-direction
		if (regType=="xGrad"):
			if(pyinfo): print("--- Horizontal gradient regularization ---")
			inv_log.addToLog("--- Horizontal gradient regularization ---")
			fat=spatialDerivModule.xGradInit(sys.argv)
			gradOp=spatialDerivModule.xGradPython(modelInit,modelInit,fat)
			gradNlOp=pyOp.NonLinearOperator(gradOp,gradOp)
			fwiProb=Prblm.ProblemL2NonLinearReg(modelInit,data,fwiInvOp,epsilon,grad_mask=maskGradient,reg_op=gradNlOp,minBound=minBoundVector,maxBound=maxBoundVector)

		# Sum of spatial gradients in z and x-directions
		if (regType=="zxGrad"):
			if(pyinfo): print("--- Gradient regularization in both directions ---")
			inv_log.addToLog("--- Gradient regularization in both directions ---")
			fat=spatialDerivModule.zxGradInit(sys.argv)
			gradOp=spatialDerivModule.zxGradPython(modelInit,modelInit,fat)
			gradNlOp=pyOp.NonLinearOperator(gradOp,gradOp)
			fwiProb=Prblm.ProblemL2NonLinearReg(modelInit,data,fwiInvOp,epsilon,grad_mask=maskGradient,reg_op=gradNlOp,minBound=minBoundVector,maxBound=maxBoundVector)

		# Evaluate Epsilon
		if (epsilonEval==1):
			if(pyinfo): print("--- Epsilon evaluation ---")
			inv_log.addToLog("--- Epsilon evaluation ---")
			epsilonOut=fwiProb.estimate_epsilon()
			if(pyinfo): print("--- Epsilon value: ",epsilonOut," ---")
			inv_log.addToLog("--- Epsilon value: %s ---"%(epsilonOut))
			quit()

	# No regularization
	else:
		fwiProb=Prblm.ProblemL2NonLinear(modelInit,data,fwiInvOp,grad_mask=maskGradient,minBound=minBoundVector,maxBound=maxBoundVector)

	############################# Solver #######################################
	# Solver
	NLCGsolver=NLCG.NLCGsolver(stop,logger=inv_log)

	# Manual step length
	initStep=parObject.getInt("initStep",-1)
	if (initStep>0):
		NLCGsolver.stepper.alpha=initStep

	# Solver
	NLCGsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	NLCGsolver.run(fwiProb,verbose=info)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

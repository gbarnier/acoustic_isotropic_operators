#!/usr/bin/env python3.6
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
import pyNLCGsolver as NLCG
import pyLBFGSsolver as LBFGS
import pyProblem as Prblm
import pyStopperBase as Stopper
import pyStepperParabolic as Stepper
import inversionUtils
from sys_util import logger

# Template for FWI workflow
if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	pyinfo=parObject.getInt("pyinfo",1)
	spline=parObject.getInt("spline",0)
	dataTaper=parObject.getInt("dataTaper",0)
	regType=parObject.getString("reg","None")
	reg=0
	if (regType != "None"): reg=1
	epsilonEval=parObject.getInt("epsilonEval",0)
	gradientMask=parObject.getInt("gradientMask",0)

	# Nonlinear solver
	solverType=parObject.getString("solver","nlcg")
	stepper=parObject.getString("stepper","default")

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)

	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------------ Conventional FWI -------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------------ Conventional FWI -------------------------")

	############################# Initialization ###############################
	# Spline
	if (spline==1):
		modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSplineModule.bSpline2dInit(sys.argv)

	# Data taper
	if (dataTaper==1):
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth=dataTaperModule.dataTaperInit(sys.argv)

	# FWI nonlinear operator
	modelFineInit,data,wavelet,parObject,sourcesVector,receiversVector=Acoustic_iso_float.nonlinearFwiOpInitFloat(sys.argv)

	# Born
	_,_,_,_,_,sourcesSignalsVector,_=Acoustic_iso_float.BornOpInitFloat(sys.argv)

	# Gradient mask
	if (gradientMask==1):
		vel,bufferUp,bufferDown,taperExp,fat,wbShift=maskGradientModule.maskGradientInit(sys.argv)

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
	# Nonlinear
	nonlinearFwiOp=Acoustic_iso_float.nonlinearFwiPropShotsGpu(modelFineInit,data,wavelet,parObject,sourcesVector,receiversVector)

	# Born
	BornOp=Acoustic_iso_float.BornShotsGpu(modelFineInit,data,modelFineInit,parObject,sourcesVector,sourcesSignalsVector,receiversVector)
	BornInvOp=BornOp
	if (gradientMask==1):
		maskGradientOp=maskGradientModule.maskGradient(modelFineInit,modelFineInit,vel,bufferUp,bufferDown,taperExp,fat,wbShift)
		BornInvOp=pyOp.ChainOperator(maskGradientOp,BornOp)

	# Conventional FWI
	fwiOp=pyOp.NonLinearOperator(nonlinearFwiOp,BornInvOp,BornOp.setVel)
	fwiInvOp=fwiOp
	modelInit=modelFineInit

	# Spline
	if (spline==1):
		if(pyinfo): print("--- Using spline interpolation ---")
		inv_log.addToLog("--- Using spline interpolation ---")
		modelInit=modelCoarseInit
		splineOp=interpBSplineModule.bSpline2d(modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp) # Create spline nonlinear operator

	# Data taper
	if (dataTaper==1):
		if(pyinfo): print("--- Using data tapering ---")
		inv_log.addToLog("--- Using data tapering ---")
		dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth)
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
	# Nonlinear conjugate gradient
	if (solverType=="nlcg"):
		nlSolver=NLCG.NLCGsolver(stop,logger=inv_log)
	# LBFGS
	elif (solverType=="lbfgs"):
		nlSolver=LBFGS.LBFGSsolver(stop,logger=inv_log)
	# Steepest descent
	elif (solverType=="sd"):
		nlSolver=NLCG.NLCGsolver(stop,beta_type="SD",logger=inv_log)

	############################# Stepper ######################################
	if (stepper == "parabolic"):
		nlSolver.stepper.eval_parab=True
	elif (stepper == "linear"):
		nlSolver.stepper.eval_parab=False
	elif (stepper == "parabolicNew"):
		print("New parabolic stepper")
		nlSolver.stepper = Stepper.ParabolicStepConst()

	####################### Manual initial step length #########################
	initStep=parObject.getInt("initStep",-1)
	if (initStep>0):
		nlSolver.stepper.alpha=initStep

	nlSolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	nlSolver.run(fwiProb,verbose=info)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

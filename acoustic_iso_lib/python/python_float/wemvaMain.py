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
import dsoGpuModule

# Solver library
import pyOperator as pyOp
import pyNLCGsolver as NLCG
import pyLBFGSsolver as LBFGS
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
from sys_util import logger

# Template for Wemva workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	pyinfo=parObject.getInt("pyinfo",1)
	spline=parObject.getInt("spline",0)
	defocusOp=parObject.getString("defocusOp","dso")
	regType=parObject.getString("reg","None")
	reg=0
	if (regType != "None"): reg=1
	epsilonEval=parObject.getInt("epsilonEval",0)
	gradientMask=parObject.getInt("gradientMask",0)

	# Nonlinear solver
	solverType=parObject.getString("solver","nlcg")
	evalParab=parObject.getInt("evalParab",1)

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)

	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------------ Conventional WEMVA -------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------------ Conventional WEMVA -------------------------")

	############################# Initialization ###############################
	# Spline
	if (spline==1):
		modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSplineModule.bSpline2dInit(sys.argv)

	# Wemva nonlinear operator (Born extended adjoint)
	modelFineInit,image,parObject,sourcesVector,sourcesSignalsVector,receiversVector,seismicData=Acoustic_iso_float.wemvaNonlinearOpInitFloat(sys.argv)

	# Wemva operator
	_,image,velocity,parObject,sourcesVector,sourcesSignalsVector,receiversVector,seismicDataWemva=Acoustic_iso_float.wemvaExtOpInitFloat(sys.argv)

	# Gradient mask
	if (gradientMask==1):
		velocity,bufferUp,bufferDown,taperExp,fat,wbShift=maskGradientModule.maskGradientInit(sys.argv)

	############################# Read files ###################################
	# Coarse-grid model
	if (spline==1):
		modelCoarseInitFile=parObject.getString("modelCoarseInit")
		modelCoarseInit=genericIO.defaultIO.getVector(modelCoarseInitFile,ndims=2)

	# Contruct a file of zeros for the "data"
	data=image.clone()
	data.scale(0.0)

	############################# Instanciation ################################
	# Nonlinear wemva (Ext Born adjoint)
	wemvaNonlinearOp=Acoustic_iso_float.wemvaNonlinearShotsGpu(modelFineInit,image,parObject.param,sourcesVector,sourcesSignalsVector,receiversVector,seismicData)

	# Wemva
	wemvaExtOp=Acoustic_iso_float.wemvaExtShotsGpu(modelFineInit,image,velocity,parObject.param,sourcesVector,sourcesSignalsVector,receiversVector,seismicDataWemva)
	wemvaExtInvOp=wemvaExtOp
	if (gradientMask==1):
		maskGradientOp=maskGradientModule.maskGradient(modelFineInit,modelFineInit,velocity,bufferUp,bufferDown,taperExp,fat,wbShift)
		wemvaExtInvOp=pyOp.ChainOperator(maskGradientOp,wemvaExtOp)

	# Wemva full operator
	wemvaOp=pyOp.NonLinearOperator(wemvaNonlinearOp,wemvaExtInvOp,wemvaExtOp.setVel)
	wemvaInvOp=wemvaOp
	modelInit=modelFineInit

	# Spline
	if (spline==1):
		if(pyinfo): print("--- Using spline interpolation ---")
		inv_log.addToLog("--- Using spline interpolation ---")
		modelInit=modelCoarseInit
		splineOp=interpBSplineModule.bSpline2d(modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp) # Create spline nonlinear operator

	# Dso
	if (defocusOp=="dso"):
		if(pyinfo): print("--- Using Dso image defocusing operator ---")
		inv_log.addToLog("--- Using Dso image defocusing operator ---")
		nz,nx,nExt,fat,dsoZeroShift=dsoGpuModule.dsoGpuInit(sys.argv)
		dsoOp=dsoGpuModule.dsoGpu(modelInit,modelInit,nz,nx,nExt,fat,dsoZeroShift)
		defocusNlOp=pyOp.NonLinearOperator(dsoOp,dsoOp) # Create dso nonlinear operator

	elif (defocusOp=="id"):
		if(pyinfo): print("--- Using identity image defocusing operator ---")
		inv_log.addToLog("--- Using identity image defocusing operator ---")

	# Concatenate operators
	if (spline==1 and defocusOp=="dso"):
		wemvaInvOpTemp=pyOp.CombNonlinearOp(splineNlOp,wemvaOp)
		wemvaInvOp=pyOp.CombNonlinearOp(wemvaInvOpTemp,defocusNlOp)
	if (spline==1 and defocusOp=="id"):
	   wemvaInvOp=pyOp.CombNonlinearOp(splineNlOp,wemvaOp)

	########################## Manual gradient mask ############################
	maskGradientFile=parObject.getString("maskGradient","NoMaskGradientFile")
	if (maskGradientFile=="NoMaskGradientFile"):
		manualMaskGradient=None
	else:
		manualMaskGradient=genericIO.defaultIO.getVector(maskGradientFile,ndims=2)

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
			wemvaProb=Prblm.ProblemL2NonLinearReg(modelInit,data,wemvaInvOp,epsilon,grad_mask=manualMaskGradient,minBound=minBoundVector,maxBound=maxBoundVector)

		# Spatial gradient in z-direction
		if (regType=="zGrad"):
			if(pyinfo): print("--- Vertical gradient regularization ---")
			inv_log.addToLog("--- Vertical gradient regularization ---")
			fat=spatialDerivModule.zGradInit(sys.argv)
			if (spline==1): fat=0
			gradOp=spatialDerivModule.zGradPython(modelInit,modelInit,fat)
			gradNlOp=pyOp.NonLinearOperator(gradOp,gradOp)
			wemvaProb=Prblm.ProblemL2NonLinearReg(modelInit,data,wemvaInvOp,epsilon,grad_mask=manualMaskGradient,reg_op=gradNlOp,minBound=minBoundVector,maxBound=maxBoundVector)

		# Spatial gradient in x-direction
		if (regType=="xGrad"):
			if(pyinfo): print("--- Horizontal gradient regularization ---")
			inv_log.addToLog("--- Horizontal gradient regularization ---")
			fat=spatialDerivModule.xGradInit(sys.argv)
			if (spline==1): fat=0
			gradOp=spatialDerivModule.xGradPython(modelInit,modelInit,fat)
			gradNlOp=pyOp.NonLinearOperator(gradOp,gradOp)
			wemvaProb=Prblm.ProblemL2NonLinearReg(modelInit,data,wemvaInvOp,epsilon,grad_mask=manualMaskGradient,reg_op=gradNlOp,minBound=minBoundVector,maxBound=maxBoundVector)

		# Sum of spatial gradients in z and x-directions
		if (regType=="zxGrad"):
			if(pyinfo): print("--- Gradient regularization in both directions ---")
			inv_log.addToLog("--- Gradient regularization in both directions ---")
			fat=spatialDerivModule.zxGradInit(sys.argv)
			if (spline==1): fat=0
			gradOp=spatialDerivModule.zxGradPython(modelInit,modelInit,fat)
			gradNlOp=pyOp.NonLinearOperator(gradOp,gradOp)
			wemvaProb=Prblm.ProblemL2NonLinearReg(modelInit,data,wemvaInvOp,epsilon,grad_mask=manualMaskGradient,reg_op=gradNlOp,minBound=minBoundVector,maxBound=maxBoundVector)

		# Evaluate Epsilon
		if (epsilonEval==1):
			if(pyinfo): print("--- Epsilon evaluation ---")
			inv_log.addToLog("--- Epsilon evaluation ---")
			epsilonOut=wemvaProb.estimate_epsilon()
			if(pyinfo): print("--- Epsilon value: ",epsilonOut," ---")
			inv_log.addToLog("--- Epsilon value: %s ---"%(epsilonOut))
			quit()

	# No regularization
	else:
		wemvaProb=Prblm.ProblemL2NonLinear(modelInit,data,wemvaInvOp,grad_mask=manualMaskGradient,minBound=minBoundVector,maxBound=maxBoundVector)

	############################# Solver #######################################
	# Solver
	# Nonlinear solver
	if (solverType=="nlcg"):
		nlSolver=NLCG.NLCGsolver(stop,logger=inv_log)
	elif (solverType=="lbfgs"):
		nlSolver=LBFGS.LBFGSsolver(stop,logger=inv_log)

	if (evalParab==0):
		nlSolver.stepper.eval_parab=False

	# Manual step length
	initStep=parObject.getInt("initStep",-1)
	if (initStep>0):
		nlSolver.stepper.alpha=initStep

	nlSolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	nlSolver.run(wemvaProb,verbose=info)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

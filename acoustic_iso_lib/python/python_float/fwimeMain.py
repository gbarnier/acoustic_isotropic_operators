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
import maskGradientModule

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyNLCGsolver as NLCG
import pyLBFGSsolver as LBFGS
import pyProblem as Prblm
import pyVPproblem as pyVp
import pyStopperBase as Stopper
from sys_util import logger
import inversionUtils

# Template for FWIME workflow
if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Auxiliary operators
	spline=parObject.getInt("spline")
	dataTaper=parObject.getInt("dataTaper")
	gradientMask=parObject.getInt("gradientMask")

	# Regularization
	regType=parObject.getString("reg")
	reg=0
	if (regType != "None"): reg=1
	epsilonEval=parObject.getInt("epsilonEval",0)

	# Nonlinear solver
	nlSolverType=parObject.getString("nlSolver")
	evalParab=parObject.getInt("evalParab")

	print("-------------------------------------------------------------------")
	print("------------------------------ FWIME ------------------------------")
	print("-------------------------------------------------------------------\n")

	############################# Initialization ###############################
	# Spline
	if (spline==1):
		print("--- Using spline interpolation ---")
		modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSplineModule.bSpline2dInit(sys.argv)

	# Data taper
	if (dataTaper==1):
		print("--- Using data tapering ---")
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth=dataTaperModule.dataTaperInit(sys.argv)

	# Gradient mask
	if (gradientMask==1):
		print("--- Using gradient masking ---")
		vel,bufferUp,bufferDown,taperExp,fat,wbShift=maskGradientModule.maskGradientInit(sys.argv)

	# Nonlinear modeling operator
	modelFineInit,data,wavelet,parObject,sourcesVector,receiversVector=Acoustic_iso_float.nonlinearFwiOpInitFloat(sys.argv)

	# Born
	_,_,_,_,_,sourcesSignalsVector,_=Acoustic_iso_float.BornOpInitFloat(sys.argv)

	# Born extended
	reflectivityExtInit,_,vel,_,_,_,_=Acoustic_iso_float.BornExtOpInitFloat(sys.argv)

	# Tomo extended
	_,_,_,_,_,_,_,_=Acoustic_iso_float.tomoExtOpInitFloat(sys.argv)

	# Dso
	nz,nx,nExt,fat,zeroShift=dsoGpuModule.dsoGpuInit(sys.argv)

	# ############################# Read files ###################################
	# The initial model is read during the initialization of the nonlinear operator (no need to re-read it)
	# Except for the waveletFile
	# Seismic source
	waveletFile=parObject.getString("sources")
	wavelet=genericIO.defaultIO.getVector(waveletFile,ndims=3)

	# Read initial extended reflectivity
	reflectivityExtInitFile=parObject.getString("reflectivityExtInit","None")
	if (reflectivityExtInitFile=="None"):
		reflectivityExtInit.scale(0.0)
	else:
		reflectivityExtInit=genericIO.defaultIO.getVector(reflectivityExtInitFile,ndims=3)

	# Coarse-grid model
	modelInit=modelFineInit
	if (spline==1):
		modelCoarseInitFile=parObject.getString("modelCoarseInit")
		modelCoarseInit=genericIO.defaultIO.getVector(modelCoarseInitFile,ndims=2)
		modelInit=modelCoarseInit

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	############################# Auxiliary operators ##########################
	if (spline==1):
		splineOp=interpBSplineModule.bSpline2d(modelInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp) # Create spline nonlinear operator

	# Data taper
	if (dataTaper==1):
		dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth)
		dataTapered=data.clone()
		dataTaperOp.forward(False,data,dataTapered) # Apply tapering to the data
		data=dataTapered
		dataTaperNlOp=pyOp.NonLinearOperator(dataTaperOp,dataTaperOp) # Create dataTaper nonlinear operator

	############################# Instanciation of g ###########################
	# Nonlinear
	nonlinearFwdOp=Acoustic_iso_float.nonlinearFwiPropShotsGpu(modelFineInit,data,wavelet,parObject,sourcesVector,receiversVector)

	# Born
	BornOp=Acoustic_iso_float.BornShotsGpu(modelFineInit,data,modelFineInit,parObject,sourcesVector,sourcesSignalsVector,receiversVector)
	BornInvOp=BornOp
	if (gradientMask==1):
		maskGradientOp=maskGradientModule.maskGradient(modelFineInit,modelFineInit,vel,bufferUp,bufferDown,taperExp,fat,wbShift)
		BornInvOp=pyOp.ChainOperator(maskGradientOp,BornOp)

	# g nonlinear (same as conventional FWI operator)
	gOp=pyOp.NonLinearOperator(nonlinearFwdOp,BornInvOp,BornOp.setVel)
	gInvOp=gOp

	# Concatenate operators
	if (spline==1 and dataTaper==0):
		gInvOp=pyOp.CombNonlinearOp(splineNlOp,gOp)
	if (spline==0 and dataTaper==1):
		gInvOp=pyOp.CombNonlinearOp(gOp,dataTaperNlOp)
	if (spline==1 and dataTaper==1):
		gInvOpTemp=pyOp.CombNonlinearOp(splineNlOp,gOp) # Combine spline and FWI
		gInvOp=pyOp.CombNonlinearOp(gInvOpTemp,dataTaperNlOp) # Combine everything

	################ Instanciation of variable projection operator #############
	# Born extended
	BornExtOp=Acoustic_iso_float.BornExtShotsGpu(reflectivityExtInit,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)
	BornExtInvOp=BornExtOp
	if (spline==1):
		BornExtOp.add_spline(splineOp)

	# Tomo
	tomoExtOp=Acoustic_iso_float.tomoExtShotsGpu(modelFineInit,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector,reflectivityExtInit)
	tomoExtInvOp=tomoExtOp

	# Concatenate operators
	if (gradientMask==1 and dataTaper==1):
		tomoTemp1=pyOp.ChainOperator(maskGradientOp,tomoExtOp)
		tomoExtInvOp=pyOp.ChainOperator(tomoTemp1,dataTaperOp)
		BornExtInvOp=pyOp.ChainOperator(BornExtOp,dataTaperOp)
	if (gradientMask==1 and dataTaper==0):
		tomoExtInvOp=pyOp.ChainOperator(maskGradientOp,tomoExtOp)
	if (gradientMask==0 and dataTaper==1):
		BornExtInvOp=pyOp.ChainOperator(BornExtOp,dataTaperOp)
		tomoExtInvOp=pyOp.ChainOperator(tomoExtOp,dataTaperOp)

	# Dso
	dsoOp=dsoGpuModule.dsoGpu(reflectivityExtInit,reflectivityExtInit,nz,nx,nExt,fat,zeroShift)

	# h nonlinear
	hNonlinearDummyOp=pyOp.ZeroOp(modelFineInit,data)
	hNonlinearOp=pyOp.NonLinearOperator(hNonlinearDummyOp,tomoExtInvOp,tomoExtOp.setVel) # We don't need the nonlinear fwd (the residuals are already computed in during the variable projection step)
	hNonlinearInvOp=hNonlinearOp
	if(spline == 1):
		hNonlinearInvOp=pyOp.CombNonlinearOp(splineNlOp,hNonlinearOp) # Combine everything

	# Variable projection operator for the data fitting term
	vpOp=pyVp.VpOperator(hNonlinearInvOp,BornExtInvOp,BornExtOp.setVel,tomoExtOp.setReflectivityExt)

	# Regularization operators
	dsoNonlinearJacobian=pyOp.ZeroOp(modelInit,reflectivityExtInit)
	dsoNonlinearDummy=pyOp.ZeroOp(modelInit,reflectivityExtInit)
	dsoNonlinearOp=pyOp.NonLinearOperator(dsoNonlinearDummy,dsoNonlinearJacobian,pyOp.dummy_set_background)

	# Variable projection operator for the regularization term
	vpRegOp=pyVp.VpOperator(dsoNonlinearOp,dsoOp,pyOp.dummy_set_background,pyOp.dummy_set_background)

	############################### solver #####################################
	# Initialize solvers
	stopNl,logFileNl,saveObjNl,saveResNl,saveGradNl,saveModelNl,invPrefixNl,bufferSizeNl,iterSamplingNl,restartFolderNl,flushMemoryNl,stopLin,logFileLin,saveObjLin,saveResLin,saveGradLin,saveModelLin,invPrefixLin,bufferSizeLin,iterSamplingLin,restartFolderLin,flushMemoryLin,epsilon,info=inversionUtils.inversionVpInit(sys.argv)

	# linear solver
	linSolver=LCG.LCGsolver(stopLin,logger=logger(logFileLin))
	linSolver.setDefaults(save_obj=saveObjLin,save_res=saveResLin,save_grad=saveGradLin,save_model=saveModelLin,prefix=invPrefixLin,iter_buffer_size=bufferSizeLin,iter_sampling=iterSamplingLin,flush_memory=flushMemoryLin)

	# Nonlinear solver
	if (nlSolverType=="nlcg"):
		nlSolver=NLCG.NLCGsolver(stopNl,logger=logger(logFileNl))
	elif(nlSolverType=="lbfgs"):
		nlSolver=LBFGS.LBFGSsolver(stopNl,logger=logger(logFileNl))
	else:
		print("**** ERROR: User did not provide a solver type ****")
		quit()

	if (evalParab==0):
		nlSolver.stepper.eval_parab=False

	# Manual step length for the nonlinear solver
	initStep=parObject.getFloat("initStep",-1)
	if (initStep>0):
		nlSolver.stepper.alpha=initStep

	nlSolver.setDefaults(save_obj=saveObjNl,save_res=saveResNl,save_grad=saveGradNl,save_model=saveModelNl,prefix=invPrefixNl,iter_buffer_size=bufferSizeNl,iter_sampling=iterSamplingNl,flush_memory=flushMemoryNl)

	############################### Bounds #####################################
	minBoundVector,maxBoundVector=Acoustic_iso_float.createBoundVectors(parObject,modelInit)

	######################### Variable projection problem ######################
	vpProb=pyVp.ProblemL2VpReg(modelInit,reflectivityExtInit,vpOp,data,linSolver,gInvOp,h_op_reg=vpRegOp,epsilon=epsilon,minBound=minBoundVector,maxBound=maxBoundVector)

	################################# Inversion ################################
	print("Run solver")
	nlSolver.run(vpProb,verbose=info)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

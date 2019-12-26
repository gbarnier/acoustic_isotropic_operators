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
import phaseOnlyXkModule

# Solver library
import pyOperator as pyOp
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as LBFGS
import pyProblem as Prblm
import pyStepper as Stepper
from pyStopper import BasicStopper as Stopper
from sys_util import logger
import inversionUtils

#Dask-related modules
from dask_util import DaskClient
import pyDaskOperator as DaskOp
import pyDaskVector

# Template for FWI workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	hostnames = parObject.getString("hostnames","noHost")
	client = None
	#Starting Dask client if requested
	if(hostnames != "noHost"):
		print("Starting Dask client using the following workers: %s"%(hostnames))
		client = DaskClient(hostnames=hostnames.split(","))
		print("Client has started!")
		nWrks = client.getNworkers()

	pyinfo=parObject.getInt("pyinfo",1)
	spline=parObject.getInt("spline",0)
	dataTaper=parObject.getInt("dataTaper",0)
	gradientMask=parObject.getInt("gradientMask",0)
	dataNormalization=parObject.getString("dataNormalization","None")
	regType=parObject.getString("reg","None")
	reg=0
	if (regType != "None"): reg=1
	epsilonEval=parObject.getInt("epsilonEval",0)

	# Nonlinear solver
	solverType=parObject.getString("solver")
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
	modelFineInit,dataInit,wavelet,parObject1,sourcesVector,receiversVector,modelFineInitLocal=Acoustic_iso_float.nonlinearFwiOpInitFloat(sys.argv,client)
	# Born
	_,_,_,_,_,sourcesSignalsVector,_,_=Acoustic_iso_float.BornOpInitFloat(sys.argv,client)

	# Gradient mask
	if (gradientMask==1):
		print("--- Using gradient masking ---")
		velLocal,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile=maskGradientModule.maskGradientInit(sys.argv)

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

	############################# Instantiation ################################

	#Dask interface
	if client:
		wavelet = pyDaskVector.DaskVector(client,vectors=[wavelet]*nWrks)
		#Spreading operator and concatenating with non-linear and born operators
		Sprd = DaskOp.DaskSpreadOp(client,modelFineInitLocal,[1]*nWrks)
		# Nonlinear
		nlOp_args = [(modelFineInit.vecDask[iwrk],dataInit.vecDask[iwrk],wavelet.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		nonlinearFwiOp = DaskOp.DaskOperator(client,Acoustic_iso_float.nonlinearFwiPropShotsGpu,nlOp_args,[1]*nWrks)
		#Concatenating spreading and non-linear
		nonlinearFwiOp = pyOp.ChainOperator(Sprd,nonlinearFwiOp)

		# Born
		BornOp_args = [(modelFineInit.vecDask[iwrk],dataInit.vecDask[iwrk],modelFineInit.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		BornOpDask = DaskOp.DaskOperator(client,Acoustic_iso_float.BornShotsGpu,BornOp_args,[1]*nWrks,setbackground_func_name="setVel",spread_op=Sprd)
		#Concatenating spreading and Born
		BornOp = pyOp.ChainOperator(Sprd,BornOpDask)

	else:
		# Nonlinear
		nonlinearFwiOp=Acoustic_iso_float.nonlinearFwiPropShotsGpu(modelFineInit,dataInit,wavelet,parObject,sourcesVector,receiversVector)

		# Born
		BornOp=Acoustic_iso_float.BornShotsGpu(modelFineInit,dataInit,modelFineInit,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

	#Born operator pointer for inversion
	BornInvOp=BornOp

	if (gradientMask==1):
		maskGradientOp=maskGradientModule.maskGradient(modelFineInitLocal,modelFineInitLocal,velLocal,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile)
		BornInvOp=pyOp.ChainOperator(maskGradientOp,BornOp)
		gMask=maskGradientOp.getMask()

	# Conventional FWI
	if client:
		fwiInvOp=pyOp.NonLinearOperator(nonlinearFwiOp,BornInvOp,BornOpDask.set_background)
	else:
		fwiInvOp=pyOp.NonLinearOperator(nonlinearFwiOp,BornInvOp,BornOp.setVel)
	modelInit=modelFineInitLocal

	# Spline
	if (spline==1):
		if(pyinfo): print("--- Using spline interpolation ---")
		inv_log.addToLog("--- Using spline interpolation ---")
		modelInit=modelCoarseInit
		splineOp=interpBSplineModule.bSpline2d(modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp) # Create spline nonlinear operator
		fwiInvOp=pyOp.CombNonlinearOp(splineNlOp,fwiInvOp)

	#Dask interface
	if(client):
		#Chunking the data and spreading them across workers if dask was requested
		data = Acoustic_iso_float.chunkData(data,BornOp.getRange())

	# Data normalization
	if (dataNormalization=="xukai"):
		if(pyinfo): print("--- Using Xukai's trace normalization ---")
		inv_log.addToLog("--- Using Xukai's trace normalization ---")
		if client:
			XkOp_args = [(data.vecDask[iwrk],data.vecDask[iwrk]) for iwrk in range(nWrks)]
			phaseOnlyXkOp = DaskOp.DaskOperator(client,phaseOnlyXkModule.phaseOnlyXk,XkOp_args,[1]*nWrks)
			XkJacOp_args = [data.vecDask[iwrk] for iwrk in range(nWrks)]
			phaseOnlyXkJacOp = DaskOp.DaskOperator(client,phaseOnlyXkModule.phaseOnlyXkJac,XkJacOp_args,[1]*nWrks,setbackground_func_name="setData")
			phaseOnlyXkNlOp=pyOp.NonLinearOperator(phaseOnlyXkOp,phaseOnlyXkJacOp,phaseOnlyXkJacOp.set_background)
		else:
			phaseOnlyXkOp=phaseOnlyXkModule.phaseOnlyXk(data,data) # Instanciate forward operator
			phaseOnlyXkJacOp=phaseOnlyXkModule.phaseOnlyXkJac(data) # Instanciate Jacobian operator
			phaseOnlyXkNlOp=pyOp.NonLinearOperator(phaseOnlyXkOp,phaseOnlyXkJacOp,phaseOnlyXkJacOp.setData) # Instantiate the nonlinear operator
		fwiInvOp=pyOp.CombNonlinearOp(fwiInvOp,phaseOnlyXkNlOp)
		obsDataNormalized=data.clone() # Apply normalization to data
		phaseOnlyXkOp.forward(False,data,obsDataNormalized)
		data=obsDataNormalized

	# Data taper
	if (dataTaper==1):
		if(pyinfo): print("--- Using data tapering ---")
		inv_log.addToLog("--- Using data tapering ---")
		if client:
			hypers = client.getClient().map(lambda x: x.getHyper(),data.vecDask,pure=False)
			dataTaper_args = [(data.vecDask[iwrk],data.vecDask[iwrk],t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,hypers[iwrk],time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth) for iwrk in range(nWrks)]
			dataTaperOp = DaskOp.DaskOperator(client,dataTaperModule.datTaper,dataTaper_args,[1]*nWrks)
		else:
			dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth)
		#Tapering observed data and constructing FWI operator
		dataTapered=data.clone()
		dataTaperOp.forward(False,data,dataTapered) # Apply tapering to the data
		data=dataTapered
		dataTaperNlOp=pyOp.NonLinearOperator(dataTaperOp,dataTaperOp) # Create dataTaper nonlinear operator
		fwiInvOp=pyOp.CombNonlinearOp(fwiInvOp,dataTaperNlOp)

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
			fwiProb=Prblm.ProblemL2NonLinearReg(modelInit,data,fwiInvOp,epsilon,grad_mask=gMask,minBound=minBoundVector,maxBound=maxBoundVector)

		# Spatial gradient in z-direction
		if (regType=="zGrad"):
			if(pyinfo): print("--- Vertical gradient regularization ---")
			inv_log.addToLog("--- Vertical gradient regularization ---")
			fat=spatialDerivModule.zGradInit(sys.argv)
			gradOp=spatialDerivModule.zGradPython(modelInit,modelInit,fat)
			gradNlOp=pyOp.NonLinearOperator(gradOp,gradOp)
			fwiProb=Prblm.ProblemL2NonLinearReg(modelInit,data,fwiInvOp,epsilon,grad_mask=gMask,reg_op=gradNlOp,minBound=minBoundVector,maxBound=maxBoundVector)

		# Spatial gradient in x-direction
		if (regType=="xGrad"):
			if(pyinfo): print("--- Horizontal gradient regularization ---")
			inv_log.addToLog("--- Horizontal gradient regularization ---")
			fat=spatialDerivModule.xGradInit(sys.argv)
			gradOp=spatialDerivModule.xGradPython(modelInit,modelInit,fat)
			gradNlOp=pyOp.NonLinearOperator(gradOp,gradOp)
			fwiProb=Prblm.ProblemL2NonLinearReg(modelInit,data,fwiInvOp,epsilon,grad_mask=gMask,reg_op=gradNlOp,minBound=minBoundVector,maxBound=maxBoundVector)

		# Sum of spatial gradients in z and x-directions
		if (regType=="zxGrad"):
			if(pyinfo): print("--- Gradient regularization in both directions ---")
			inv_log.addToLog("--- Gradient regularization in both directions ---")
			fat=spatialDerivModule.zxGradInit(sys.argv)
			gradOp=spatialDerivModule.zxGradPython(modelInit,modelInit,fat)
			gradNlOp=pyOp.NonLinearOperator(gradOp,gradOp)
			fwiProb=Prblm.ProblemL2NonLinearReg(modelInit,data,fwiInvOp,epsilon,grad_mask=gMask,reg_op=gradNlOp,minBound=minBoundVector,maxBound=maxBoundVector)

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
		fwiProb=Prblm.ProblemL2NonLinear(modelInit,data,fwiInvOp,minBound=minBoundVector,maxBound=maxBoundVector)

	############################# Solver #######################################
	# Nonlinear conjugate gradient
	if (solverType=="nlcg"):
		nlSolver=NLCG(stop,logger=inv_log)
	# LBFGS
	elif (solverType=="lbfgs"):
		nlSolver=LBFGS(stop,logger=inv_log)
	# Steepest descent
	elif (solverType=="sd"):
		nlSolver=NLCG(stop,beta_type="SD",logger=inv_log)

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

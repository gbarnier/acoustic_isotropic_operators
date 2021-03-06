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
import maskGradientModule
import spatialDerivModule

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
from pyLinearSolver import LSQRsolver as LSQR
import pyProblem as Prblm
import inversionUtils
from sys_util import logger

#Dask-related modules
import pyDaskOperator as DaskOp
import pyDaskVector

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Checking if Dask was requested
	client, nWrks = Acoustic_iso_float.create_client(parObject)

	pyinfo=parObject.getInt("pyinfo",1)
	spline=parObject.getInt("spline",0)
	dataTaper=parObject.getInt("dataTaper",0)
	gradientMask=parObject.getInt("gradientMask",0)
	regType=parObject.getString("reg","None")
	solver=parObject.getString("solver","LCG") #[LCG,LSQR]
	reg=0
	if (regType != "None"): reg=1
	epsilonEval=parObject.getInt("epsilonEval",0)
	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------ Conventional linearized inversion --------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ Conventional linearized inversion --------------")

	############################# Initialization ###############################
	# Spline
	if (spline==1):
		modelCoarseInit,modelFineInit,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSplineModule.bSpline2dInit(sys.argv)

	# Data taper
	if (dataTaper==1):
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth,streamers=dataTaperModule.dataTaperInit(sys.argv)

	# Born arguments
	modelFineInit,data,vel,parObject1,sourcesVector,sourcesSignalsVector,receiversVector,modelFineInitLocal=Acoustic_iso_float.BornOpInitFloat(sys.argv,client)
	# Born operator
	if client:
		#Instantiating Dask Operator
		BornOp_args = [(modelFineInit.vecDask[iwrk],data.vecDask[iwrk],vel[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		BornOp = DaskOp.DaskOperator(client,Acoustic_iso_float.BornShotsGpu,BornOp_args,[1]*nWrks)
		#Adding spreading operator and concatenating with Born operator (using modelFineInitLocal)
		Sprd = DaskOp.DaskSpreadOp(client,modelFineInitLocal,[1]*nWrks)
		invOp = pyOp.ChainOperator(Sprd,BornOp)
	else:
		BornOp=Acoustic_iso_float.BornShotsGpu(modelFineInit,data,vel,parObject1,sourcesVector,sourcesSignalsVector,receiversVector)
		invOp=BornOp

	# Gradient mask
	if (gradientMask==1):
		print("--- Using gradient masking ---")
		velLocal,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile=maskGradientModule.maskGradientInit(sys.argv)

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
			modelInit=modelFineInitLocal.clone()
			modelInit.scale(0.0)

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	#Dask interface
	if(client):
		#Chunking the data and spreading them across workers if dask was requested
		data = Acoustic_iso_float.chunkData(data,BornOp.getRange())

	# Diagonal Preconditioning
	PrecFile = parObject.getString("PrecFile","None")
	Precond = None
	if PrecFile != "None":
		if(pyinfo): print("--- Using diagonal preconditioning ---")
		inv_log.addToLog("--- Using diagonal preconditioning ---")
		PrecVec=genericIO.defaultIO.getVector(PrecFile)
		if not PrecVec.checkSame(modelInit):
			raise ValueError("ERROR! Preconditioning diagonal inconsistent with model vector")
		Precond = pyOp.DiagonalOp(PrecVec)

	############################# Instanciation ################################

	# Spline
	if (spline==1):
		if(pyinfo): print("--- Using spline interpolation ---")
		inv_log.addToLog("--- Using spline interpolation ---")
		splineOp=interpBSplineModule.bSpline2d(modelCoarseInit,modelFineInitLocal,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)

	# Data taper
	if (dataTaper==1):
		if(pyinfo): print("--- Using data tapering ---")
		inv_log.addToLog("--- Using data tapering ---")
		if client:
			hypers = client.getClient().map(lambda x: x.getHyper(),data.vecDask,pure=False)
			dataTaper_args = [(data.vecDask[iwrk],data.vecDask[iwrk],t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,hypers[iwrk],time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth,streamers) for iwrk in range(nWrks)]
			dataTaperOp = DaskOp.DaskOperator(client,dataTaperModule.datTaper,dataTaper_args,[1]*nWrks)
		else:
			dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth,streamers)
		dataTapered=data.clone()
		dataTaperOp.forward(False,data,dataTapered) # Apply tapering to the data
		data=dataTapered


	if (gradientMask==1):
		maskGradientOp=maskGradientModule.maskGradient(modelFineInitLocal,modelFineInitLocal,velLocal,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile)
		invOp=pyOp.ChainOperator(maskGradientOp,invOp)

	# Concatenate operators
	if (spline==1 and dataTaper==0):
		invOp=pyOp.ChainOperator(splineOp,invOp)
	if (spline==0 and dataTaper==1):
		invOp=pyOp.ChainOperator(invOp,dataTaperOp)
	if (spline==1 and dataTaper==1):
		invOpTemp=pyOp.ChainOperator(splineOp,invOp)
		invOp=pyOp.ChainOperator(invOpTemp,dataTaperOp)

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
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,prec=Precond)

		# Spatial gradient in z-direction
		if (regType=="zGrad"):
			if(pyinfo): print("--- Vertical gradient regularization ---")
			inv_log.addToLog("--- Vertical gradient regularization ---")
			fat=spatialDerivModule.zGradInit(sys.argv)
			gradOp=spatialDerivModule.zGradPython(modelInit,modelInit,fat)
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,prec=Precond)

		# Spatial gradient in x-direction
		if (regType=="xGrad"):
			if(pyinfo): print("--- Horizontal gradient regularization ---")
			inv_log.addToLog("--- Horizontal gradient regularization ---")
			fat=spatialDerivModule.xGradInit(sys.argv)
			gradOp=spatialDerivModule.xGradPython(modelInit,modelInit,fat)
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,reg_op=gradOp,prec=Precond)

		# Sum of spatial gradients in z and x-directions
		if (regType=="zxGrad"):
			if(pyinfo): print("--- Gradient regularization in both directions ---")
			inv_log.addToLog("--- Gradient regularization in both directions ---")
			fat=spatialDerivModule.zxGradInit(sys.argv)
			gradOp=spatialDerivModule.zxGradPython(modelInit,modelInit,fat)
			invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,reg_op=gradOp,prec=Precond)

		# Evaluate Epsilon
		if (epsilonEval==1):
			if(pyinfo): print("--- Epsilon evaluation ---")
			inv_log.addToLog("--- Epsilon evaluation ---")
			epsilonOut=invProb.estimate_epsilon()
			if(pyinfo): print("--- Epsilon value: ",epsilonOut," ---")
			inv_log.addToLog("--- Epsilon value: %s ---"%(epsilonOut))
			quit()

	# No regularization
	else:
		invProb=Prblm.ProblemL2Linear(modelInit,data,invOp,prec=Precond)

	############################## Solver ######################################
	# Solver
	if solver == "LCG":
		Linsolver=LCG(stop,logger=inv_log)
	elif solver == "LSQR":
		Linsolver=LSQR(stop,logger=inv_log)
	else:
		raise ValueError("Unknown solver: %s"%(solver))
	Linsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	Linsolver.run(invProb,verbose=True)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

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
import dsoGpuModule
import dsoInvGpuModule

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
from pyNonLinearSolver import LBFGSsolver as LBFGS
import pyProblem as Prblm
from sys_util import logger
import inversionUtils

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
	solver=parObject.getString("solver","LCG")
	spline=parObject.getInt("spline",0)
	dataTaper=parObject.getInt("dataTaper",0)
	gradientMask=parObject.getInt("gradientMask",0)
	regType=parObject.getString("reg","None")
	reg=0
	if (regType != "None"): reg=1
	epsilonEval=parObject.getInt("epsilonEval",0)

	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("-------------------- Extended linearized inversion ----------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("-------------------- Extended linearized inversion ----------------")

	############################# Initialization ###############################
	# Spline
	if (spline==1):
		modelCoarseInit,modelFineInit,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat=interpBSplineModule.bSpline3dInit(sys.argv)

	# Data taper
	if (dataTaper==1):
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth,streamers=dataTaperModule.dataTaperInit(sys.argv)

	# Born extended
	modelFineInit,data,vel,parObject1,sourcesVector,sourcesSignalsVector,receiversVector,modelFineInitLocal=Acoustic_iso_float.BornExtOpInitFloat(sys.argv,client)
	# Born operator
	if client:
		#Instantiating Dask Operator
		BornExtOp_args = [(modelFineInit.vecDask[iwrk],data.vecDask[iwrk],vel[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		BornExtOp = DaskOp.DaskOperator(client,Acoustic_iso_float.BornExtShotsGpu,BornExtOp_args,[1]*nWrks)
		#Adding spreading operator and concatenating with Born operator (using modelFineInitLocal)
		Sprd = DaskOp.DaskSpreadOp(client,modelFineInitLocal,[1]*nWrks)
		invOp = pyOp.ChainOperator(Sprd,BornExtOp)
	else:
		BornExtOp=Acoustic_iso_float.BornExtShotsGpu(modelFineInit,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)
		invOp=BornExtOp

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
			modelInit=genericIO.defaultIO.getVector(modelInitFile,ndims=3)

	else:
		if (modelInitFile=="None"):
			modelInit=modelFineInitLocal.clone()
			modelInit.scale(0.0)
		else:
			modelInit=genericIO.defaultIO.getVector(modelInitFile,ndims=3)

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	#Dask interface
	if(client):
		#Chunking the data and spreading them across workers if dask was requested
		data = Acoustic_iso_float.chunkData(data,BornExtOp.getRange())

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

	############################# Instantiation ################################

	# Spline
	if (spline==1):
		if(pyinfo): print("--- Using spline interpolation ---")
		inv_log.addToLog("--- Using spline interpolation ---")
		splineOp=interpBSplineModule.bSpline3d(modelCoarseInit,modelFineInitLocal,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat)

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

	if (regType != "dsoPrec"):
		# Concatenate operators
		if (spline==1 and dataTaper==0):
			invOp=pyOp.ChainOperator(splineOp,invOp)
		if (spline==0 and dataTaper==1):
			invOp=pyOp.ChainOperator(invOp,dataTaperOp)
		if (spline==1 and dataTaper==1):
			invOpTemp=pyOp.ChainOperator(splineOp,invOp)
			invOp=pyOp.ChainOperator(invOpTemp,dataTaperOp)

	# Preconditioning with DSO inverse
	if (reg==1 and regType=="dsoPrec"):

		# Instanciate Dso inverse operator
		# If we also use spline, the sequence is Born(Spline(DsoInverse(p)))
		nz,nx,nExt,fat,dsoZeroShift=dsoInvGpuModule.dsoInvGpuInit(sys.argv)
		nz=modelInit.getHyper().axes[0].n
		nx=modelInit.getHyper().axes[1].n
		nExt=modelInit.getHyper().axes[2].n
		fat=0
		dsoInvOp=dsoInvGpuModule.dsoInvGpu(modelInit,modelInit,nz,nx,nExt,fat,dsoZeroShift)

		# Concatenate operators
		if (spline==1 and dataTaper==0):
			splineTempOp=pyOp.ChainOperator(dsoInvOp,splineOp)
			invOp=pyOp.ChainOperator(splineTempOp,invOp)
		if (spline==0 and dataTaper==1):
			invOpTemp=pyOp.ChainOperator(dsoInvOp,invOp)
			invOp=pyOp.ChainOperator(invOpTemp,dataTaperOp)
		if (spline==1 and dataTaper==1):
			splineTempOp=pyOp.ChainOperator(dsoInvOp,splineOp)
			invOpTemp=pyOp.ChainOperator(splineTempOp,invOp)
			invOp=pyOp.ChainOperator(invOpTemp,dataTaperOp)

	if solver == "lbfgs":
		#Solving linear problem with BFGS
		invOp = pyOp.NonLinearOperator(invOp,invOp)

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
			if solver == "lbfgs":
				invProb=Prblm.ProblemL2NonLinearReg(modelInit,data,invOp,epsilon)
			else:
				invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,prec=Precond)

		# Dso
		elif (regType=="dso"):
			if(pyinfo): print("--- DSO regularization ---")
			inv_log.addToLog("--- DSO regularization ---")

			if (spline==1):
				# If using spline, the model dimensions change, you apply the DSO on the coarse grid
				nz,nx,nExt,fat,dsoZeroShift=dsoGpuModule.dsoGpuInit(sys.argv)
				nz=modelInit.getHyper().axes[0].n
				nx=modelInit.getHyper().axes[1].n
				nExt=modelInit.getHyper().axes[2].n
				fat=0
				dsoOp=dsoGpuModule.dsoGpu(modelInit,modelInit,nz,nx,nExt,fat,dsoZeroShift)
			else:
				nz,nx,nExt,fat,dsoZeroShift=dsoGpuModule.dsoGpuInit(sys.argv)
				dsoOp=dsoGpuModule.dsoGpu(modelInit,modelInit,nz,nx,nExt,fat,dsoZeroShift)
			if solver == "lbfgs":
				invProb=Prblm.ProblemL2NonLinearReg(modelInit,data,invOp,epsilon,reg_op=pyOp.NonLinearOperator(dsoOp,dsoOp))
			else:
				invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,reg_op=dsoOp,prec=Precond)

		elif (regType=="dsoPrec"):
			if(pyinfo): print("--- DSO regularization with preconditioning ---")
			inv_log.addToLog("--- DSO regularization with preconditioning ---")
			if solver == "lbfgs":
				invProb=Prblm.ProblemL2NonLinearReg(modelInit,data,invOp,epsilon,prec=Precond)
			else:
				invProb=Prblm.ProblemL2LinearReg(modelInit,data,invOp,epsilon,prec=Precond)

		else:
			if(pyinfo): print("--- Regularization that you have required is not supported by our code ---")
			quit()

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
		if solver == "lbfgs":
			invProb=Prblm.ProblemL2NonLinear(modelInit,data,invOp)
		else:
			invProb=Prblm.ProblemL2Linear(modelInit,data,invOp,prec=Precond)

	############################## Solver ######################################
	# Solver
	if solver == "LCG":
		linSolver=LCG(stop,logger=inv_log)
	elif solver == "lbfgs":
		linSolver=LBFGS(stop,logger=inv_log,H0=Precond)
	else:
		raise ValueError("Provided requested solver (%s) not supported!"%(solver))

	linSolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	linSolver.run(invProb,verbose=True)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

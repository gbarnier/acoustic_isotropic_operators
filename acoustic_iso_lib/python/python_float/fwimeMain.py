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
import maskGradientModule

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
from pyNonLinearSolver import NLCGsolver as NLCG
from pyNonLinearSolver import LBFGSsolver as LBFGS
import pyProblem as Prblm
import pyVPproblem as pyVp
from sys_util import logger
import inversionUtils

#Dask-related modules
from dask_util import DaskClient
import pyDaskOperator as DaskOp
import pyDaskVector
import dask.distributed as daskD

# Template for FWIME workflow
if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	hostnames = parObject.getString("hostnames","noHost")
	client = None
	#Starting Dask client if requested
	if(hostnames != "noHost"):
		print("Starting Dask client using following workers: %s"%(hostnames))
		client = DaskClient(hostnames.split(","))
		print("Client has started!")
		nWrks = client.getNworkers()

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
	evalParab=parObject.getInt("evalParab",1)

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
		vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile=maskGradientModule.maskGradientInit(sys.argv,client)

	# Nonlinear modeling operator
	modelFineInit,dataInit,wavelet,parObject1,sourcesVector,receiversVector,modelFineInitLocal=Acoustic_iso_float.nonlinearFwiOpInitFloat(sys.argv,client)

	# Born
	_,_,_,_,_,sourcesSignalsVector,_,_=Acoustic_iso_float.BornOpInitFloat(sys.argv,client)

	# Born extended
	reflectivityExtInit,_,vel,_,_,_,_,reflectivityExtInitLocal=Acoustic_iso_float.BornExtOpInitFloat(sys.argv,client)

	# Tomo extended
	# _,_,_,_,_,_,_,_=Acoustic_iso_float.tomoExtOpInitFloat(sys.argv,client)

	# Dso
	nz,nx,nExt,fat,zeroShift=dsoGpuModule.dsoGpuInit(sys.argv,client)

	# ############################# Read files ###################################
	# The initial model is read during the initialization of the nonlinear operator (no need to re-read it)
	# Except for the waveletFile
	# Seismic source
	waveletFile=parObject.getString("sources")
	wavelet=genericIO.defaultIO.getVector(waveletFile,ndims=3)

	# Read initial extended reflectivity
	reflectivityExtInitFile=parObject.getString("reflectivityExtInit","None")
	if (reflectivityExtInitFile=="None"):
		reflectivityExtInitLocal.set(0.0)
	else:
		reflectivityExtInitLocal=genericIO.defaultIO.getVector(reflectivityExtInitFile,ndims=3)

	# Coarse-grid model
	modelInit=modelFineInitLocal
	if (spline==1):
		modelCoarseInitFile=parObject.getString("modelCoarseInit")
		modelCoarseInit=genericIO.defaultIO.getVector(modelCoarseInitFile,ndims=2)
		modelInit=modelCoarseInit

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)


	############################# Instanciation of g ###########################
	#Dask interface
	if client:
		wavelet = pyDaskVector.DaskVector(client,vectors=[wavelet]*nWrks)
		#Spreading operator and concatenating with non-linear and born operators
		Sprd = DaskOp.DaskSpreadOp(client,modelFineInitLocal,[1]*nWrks)
		# Nonlinear
		nlOp_args = [(modelFineInit.vecDask[iwrk],dataInit.vecDask[iwrk],wavelet.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		nonlinearFwdOp = DaskOp.DaskOperator(client,Acoustic_iso_float.nonlinearFwiPropShotsGpu,nlOp_args,[1]*nWrks)
		#Concatenating spreading and non-linear
		nonlinearFwdOp = pyOp.ChainOperator(Sprd,nonlinearFwdOp)

		# Born
		BornOp_args = [(modelFineInit.vecDask[iwrk],dataInit.vecDask[iwrk],modelFineInit.vecDask[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		BornOpDask = DaskOp.DaskOperator(client,Acoustic_iso_float.BornShotsGpu,BornOp_args,[1]*nWrks,setbackground_func_name="setVel",spread_op=Sprd)
		#Concatenating spreading and Born
		BornOp = pyOp.ChainOperator(Sprd,BornOpDask)
	else:
		# Nonlinear
		nonlinearFwdOp=Acoustic_iso_float.nonlinearFwiPropShotsGpu(modelFineInit,dataInit,wavelet,parObject,sourcesVector,receiversVector)

		# Born
		BornOp=Acoustic_iso_float.BornShotsGpu(modelFineInit,dataInit,modelFineInit,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

	BornInvOp=BornOp
	if (gradientMask==1):
		maskGradientOp=maskGradientModule.maskGradient(modelFineInitLocal,modelFineInitLocal,vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile)
		BornInvOp=pyOp.ChainOperator(maskGradientOp,BornOp)

	# g nonlinear (same as conventional FWI operator)
	if client:
		gOp=pyOp.NonLinearOperator(nonlinearFwdOp,BornInvOp,BornOpDask.set_background)
	else:
		gOp=pyOp.NonLinearOperator(nonlinearFwdOp,BornInvOp,BornOp.setVel)
	gInvOp=gOp

	#Dask interface
	if(client):
		#Chunking the data and spreading them across workers if dask was requested
		data = Acoustic_iso_float.chunkData(data,BornOp.getRange())

	############################# Auxiliary operators ##########################
	if (spline==1):
		splineOp=interpBSplineModule.bSpline2d(modelInit,modelFineInitLocal,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp) # Create spline nonlinear operator

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
			dataTapered=data.clone()
		dataTaperOp.forward(False,data,dataTapered) # Apply tapering to the data
		data=dataTapered
		dataTaperNlOp=pyOp.NonLinearOperator(dataTaperOp,dataTaperOp) # Create dataTaper nonlinear operator

	# Concatenate operators
	if (spline==1 and dataTaper==0):
		gInvOp=pyOp.CombNonlinearOp(splineNlOp,gOp)
	if (spline==0 and dataTaper==1):
		gInvOp=pyOp.CombNonlinearOp(gOp,dataTaperNlOp)
	if (spline==1 and dataTaper==1):
		gInvOpTemp=pyOp.CombNonlinearOp(splineNlOp,gOp) # Combine spline and FWI
		gInvOp=pyOp.CombNonlinearOp(gInvOpTemp,dataTaperNlOp) # Combine everything

	################ Instantiation of variable projection operator #############

	if client:
		#Instantiating Dask Operator
		BornExtOp_args = [(reflectivityExtInit.vecDask[iwrk],data.vecDask[iwrk],vel[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		BornExtOp = DaskOp.DaskOperator(client,Acoustic_iso_float.BornExtShotsGpu,BornExtOp_args,[1]*nWrks,setbackground_func_name="setVel",spread_op=Sprd)
		#Adding spreading operator and concatenating with Born operator (using modelFineInitLocal)
		BornExtInvOp = pyOp.ChainOperator(Sprd,BornExtOp)

		#Adding Spline to Born set_vel functions
		if (spline==1):
			splineOp_args = [(modelInit.vecDask[iwrk],modelFineInit.vecDask[iwrk],zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat) for iwrk in range(nWrks)]
			splineOpD = DaskOp.DaskOperator(client,interpBSplineModule.bSpline2d,splineOp_args,[1]*nWrks)
			add_spline_ftr = []
			for idx,spln_op in enumerate(splineOpD.dask_ops):
				add_spline_ftr.append(client.getClient().submit(lambda obj: obj.add_spline(spln_op)),BornExtOp.dask_ops[idx],pure=False)
			daskD.wait(add_spline_ftr)

		#Instantiating Dask Operator
		SprdRefl = DaskOp.DaskSpreadOp(client,reflectivityExtInitLocal,[1]*nWrks)
		tomoExtOp_args = [(modelFineInit.vecDask[iwrk],data.vecDask[iwrk],vel[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsVector[iwrk],receiversVector[iwrk],reflectivityFloat.vecDask[iwrk]) for iwrk in range(nWrks)]
		tomoExtOp = DaskOp.DaskOperator(client,Acoustic_iso_float.tomoExtShotsGpu,tomoExtOp_args,[1]*nWrks,setbackground_func_name="setVel",spread_op=Sprd,set_aux_name="setReflectivityExt",spread_op_aux=SprdRefl)
		#Adding spreading operator and concatenating with Born operator (using modelFloatLocal)
		tomoExtOp = pyOp.ChainOperator(Sprd,tomoExtOp)
	else:
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
	dsoOp=dsoGpuModule.dsoGpu(reflectivityExtInitLocal,reflectivityExtInitLocal,nz,nx,nExt,fat,zeroShift)

	# h nonlinear
	hNonlinearDummyOp=pyOp.ZeroOp(modelFineInitLocal,data)
	if client:
		hNonlinearOp=pyOp.NonLinearOperator(hNonlinearDummyOp,tomoExtInvOp,tomoExtOp.set_background)
	else:
		hNonlinearOp=pyOp.NonLinearOperator(hNonlinearDummyOp,tomoExtInvOp,tomoExtOp.setVel) # We don't need the nonlinear fwd (the residuals are already computed in during the variable projection step)
	hNonlinearInvOp=hNonlinearOp
	if(spline == 1):
		hNonlinearInvOp=pyOp.CombNonlinearOp(splineNlOp,hNonlinearOp) # Combine everything

	# Variable projection operator for the data fitting term
	if client:
		vpOp=pyVp.VpOperator(hNonlinearInvOp,BornExtInvOp,BornExtOp.set_background,tomoExtOp.set_aux)
	else:
		vpOp=pyVp.VpOperator(hNonlinearInvOp,BornExtInvOp,BornExtOp.setVel,tomoExtOp.setReflectivityExt)

	# Regularization operators
	dsoNonlinearJacobian=pyOp.ZeroOp(modelInit,reflectivityExtInitLocal)
	dsoNonlinearDummy=pyOp.ZeroOp(modelInit,reflectivityExtInitLocal)
	dsoNonlinearOp=pyOp.NonLinearOperator(dsoNonlinearDummy,dsoNonlinearJacobian)

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
		nlSolver=NLCG(stopNl,logger=logger(logFileNl))
		if (evalParab==0):
			nlSolver.stepper.eval_parab=False
	elif(nlSolverType=="lbfgs"):
		# By default, Lbfgs uses MT stepper
		nlSolver=LBFGS(stopNl,logger=logger(logFileNl))
	else:
		print("**** ERROR: User did not provide a nonlinear solver type ****")
		quit()

	# Manual step length for the nonlinear solver
	initStep=parObject.getFloat("initStep",-1)
	if (initStep>0):
		nlSolver.stepper.alpha=initStep

	nlSolver.setDefaults(save_obj=saveObjNl,save_res=saveResNl,save_grad=saveGradNl,save_model=saveModelNl,prefix=invPrefixNl,iter_buffer_size=bufferSizeNl,iter_sampling=iterSamplingNl,flush_memory=flushMemoryNl)

	############################### Bounds #####################################
	minBoundVector,maxBoundVector=Acoustic_iso_float.createBoundVectors(parObject,modelInit)

	######################### Variable projection problem ######################
	vpProb=pyVp.ProblemL2VpReg(modelInit,reflectivityExtInitLocal,vpOp,data,linSolver,gInvOp,h_op_reg=vpRegOp,epsilon=epsilon,minBound=minBoundVector,maxBound=maxBoundVector)

	################################# Inversion ################################
	print("Run solver")
	nlSolver.run(vpProb,verbose=info)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

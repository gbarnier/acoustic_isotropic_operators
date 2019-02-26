#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import interpBSpline2dModule
import dataTaperModule
import numpy as np
import time
import sys

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

	############ Case #1: Conventional linearized Born inversion ###############
	if (spline==0 and dataTaper==0):

		print("-------------------------------------------------------------------")
		print("------------------ Conventional linearized inversion --------------")
		print("-------------------------------------------------------------------\n")

		# Born initialization
		modelStart,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornOpInitFloat(sys.argv)

		# Read starting model
		modelStartFile=parObject.getString("modelStart")
		modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=2)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Born instanciation
		BornOp=Acoustic_iso_float.BornShotsGpu(modelStart,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

		# Problem
		lsrtmProb=Prblm.ProblemL2Linear(modelStart,data,BornOp)
	############################################################################

	######### Case #2: Linearized Born inversion with data tapering ############
	elif (spline==0 and dataTaper==1):

		print("-------------------------------------------------------------------")
		print("--------------- Linearized inversion + data tapering --------------")
		print("-------------------------------------------------------------------\n")

		# Born initialization
		modelStart,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornOpInitFloat(sys.argv)

		# Data taper initialization
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset=dataTaperModule.dataTaperInit(sys.argv)

		# Read starting model
		modelStartFile=parObject.getString("modelStart")
		modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=2)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Born instanciation
		BornOp=Acoustic_iso_float.BornShotsGpu(modelStart,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

		# Data taper instanciation
		dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset)
		dataTapered=data.clone()
		dataTaperOp.forward(False,data,dataTapered)
		data=dataTapered

		# Operator chain
		lsrtmOp=pyOp.ChainOperator(BornOp,dataTaperOp)

		# Problem
		lsrtmProb=Prblm.ProblemL2Linear(modelStart,data,lsrtmOp)

	############################################################################

	####### Case #3: Linearized Born inversion with spline interpolation #######
	elif (spline==1 and dataTaper==0):

		print("-------------------------------------------------------------------")
		print("------------ Linearized inversion + spline interpolation ----------")
		print("-------------------------------------------------------------------\n")

		# Born initialization
		modelStartFine,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornOpInitFloat(sys.argv)

		# Spline initialization
		modelStart,modelFineStart,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dModule.bSpline2dInit(sys.argv)

		# Read starting model
		modelStartFile=parObject.getString("modelStart")
		modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=2)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Born instanciation
		BornOp=Acoustic_iso_float.BornShotsGpu(modelStartFine,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

		# Spline instanciation
		splineOp=interpBSpline2dModule.bSpline2d(modelStart,modelStartFine,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)

		# Operator chain
		lsrtmOp=pyOp.ChainOperator(splineOp,BornOp)

		# Problem
		lsrtmProb=Prblm.ProblemL2Linear(modelStart,data,lsrtmOp)

	############################################################################

	########## Case #4: Linearized Born inversion spline + data taper ##########
	else:

		print("-------------------------------------------------------------------")
		print("------------ Linearized inversion + spline and data taper ---------")
		print("-------------------------------------------------------------------\n")

		# Born initialization
		modelStartFine,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornOpInitFloat(sys.argv)

		# Spline initialization
		modelStart,modelFineStart,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dModule.bSpline2dInit(sys.argv)

		# Data taper initialization
		t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset=dataTaperModule.dataTaperInit(sys.argv)

		# Read starting model
		modelStartFile=parObject.getString("modelStart")
		modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=2)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Born instanciation
		BornOp=Acoustic_iso_float.BornShotsGpu(modelStartFine,data,vel,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

		# Spline instanciation
		splineOp=interpBSpline2dModule.bSpline2d(modelStart,modelStartFine,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)

		# Data taper instanciation
		dataTaperOp=dataTaperModule.datTaper(data,data,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,data.getHyper(),time,offset)
		dataTapered=data.clone()
		dataTaperOp.forward(False,data,dataTapered)
		data=dataTapered

		# Operator chain
		lsrtmOp1=pyOp.ChainOperator(splineOp,BornOp)
		lsrtmOp=pyOp.ChainOperator(lsrtmOp1,dataTaperOp)

		# Problem
		lsrtmProb=Prblm.ProblemL2Linear(modelStart,data,lsrtmOp)

	############################################################################

	############################## Solver ######################################
	# Stopper
	stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

	# Solver
	logFile=parObject.getString("logFile")
	invPrefix=parObject.getString("prefix")
	LCGsolver=LCG.LCGsolver(stop,logger=logger(logFile))
	LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

	# Run solver
	LCGsolver.run(lsrtmProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

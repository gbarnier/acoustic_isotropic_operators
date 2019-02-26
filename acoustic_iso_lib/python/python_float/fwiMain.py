#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import interpBSpline2dModule
import numpy as np
import time
import sys

# Solver library
import pyOperator as pyOp
import pyNLCGsolver as NLCG
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger

# Template for FWI workflow
if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	par=ioDef.getParamObj()
    spline=par.getString("spline",0)

	print("-------------------------------------------------------------------")
	print("---------------------- Running conventional FWI -------------------")
	print("-------------------------------------------------------------------\n")

	############################# Initialization ###############################
	# Nonlinear
	wavelet,data,modelStartFine,parObject,sourcesVector,receiversVector=Acoustic_iso_float.nonlinearOpInitFloat(sys.argv)

	# Born
	_,_,_,_,_,sourcesSignalsVector,_=Acoustic_iso_float.BornOpInitFloat(sys.argv)

    if (spline==1):
    	modelStartCoarse,_,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dModule.bSpline2dInit(sys.argv)

	############################# Read files ###################################
	# Seismic source
	waveletFile=parObject.getString("sources")
	wavelet=genericIO.defaultIO.getVector(waveletFile,ndims=3)

	# Data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=3)

	############################# Instanciation ################################
	# Nonlinear
	nonlinearVelocityOp=Acoustic_iso_float.nonlinearVelocityPropShotsGpu(modelStartFine,data,wavelet,parObject,sourcesVector,receiversVector)

	# Born
	BornOp=Acoustic_iso_float.BornShotsGpu(modelStartFine,data,modelStartFine,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

	# FWI
	fwiOp=pyOp.NonLinearOperator(nonlinearVelocityOp,BornOp,BornOp.setVel)
	fwiInvOp=fwiOp

	# Spline instanciation
	if (spline==1):
		splineOp=interpBSpline2dModule.bSpline2d(modelStartCoarse,modelStartFine,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)
		splineOp.adjoint(False,modelStartCoarse,modelStartFine)
		splineNlOp=pyOp.NonLinearOperator(splineOp,splineOp,pyOp.dummy_set_background)
		fwiInvOp=pyOp.CombNonlinearOp(splineNlOp,fwiOp)

	############################# Solver #######################################
	# L2-norm nonlinear problem
	minVal=parObject.getFloat("minBound",0.0) # Set to 0 [km/s] by default
	maxVal=parObject.getFloat("maxBound",10.0) # Set to [10km/s] by default
	minBound=modelStartFwi.clone()
	maxBound=modelStartFwi.clone()
	minBound.set(minVal)
	maxBound.set(maxVal)
	fwiProb=Prblm.ProblemL2NonLinear(modelStartFwi,data,fwiInvOp,minBound=minBound,maxBound=maxBound)
	# fwiProb=Prblm.ProblemL2NonLinear(modelStartFwi,data,fwiOp)
	# Stopper
	stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

	# Solver
	logFile=parObject.getString("logFile")
	invPrefix=parObject.getString("prefix")
	NLCGsolver=NLCG.NLCGsolver(stop,logger=logger(logFile))
	NLCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

	# Run solver
	NLCGsolver.run(fwiProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

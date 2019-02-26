#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import interpBSpline2dModule
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
	interpOp=parObject.getInt("interpOp","spline")

	# Spline interpolation
    if (spline==1):

    	# Spline initialization
    	model,data,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dModule.bSpline2dInit(sys.argv)

    	# Read starting model
    	modelStart=parObject.getString("modelStart")
    	modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=2)

    	# Read data
    	dataFile=parObject.getString("data")
    	data=genericIO.defaultIO.getVector(dataFile,ndims=2)

        # Spline instanciation
    	splineOp=interpBSpline2dModule.bSpline2d(modelStart,data,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)
    	invOp=splineOp

    	# Problem
    	invProb=Prblm.ProblemL2Linear(modelStart,data,invOp)

	############################## Solver ######################################
	# Stopper
	stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

	# Solver
	logFile=parObject.getString("logFile")
	invPrefix=parObject.getString("prefix")
	LCGsolver=LCG.LCGsolver(stop,logger=logger(logFile))
	LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

	# Run solver
	LCGsolver.run(invProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

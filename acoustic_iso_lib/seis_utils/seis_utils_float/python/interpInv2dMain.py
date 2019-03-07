#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import interpBSpline2dModule
import numpy as np
import time
import sys
import os

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger

# Template for interpolation optimization (to find a coarse model parameters given a fine model)
if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	interp=parObject.getString("interp","spline")

	# Spline interpolation
	if (interp=="spline"):

		# Spline initialization
		model,data,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dModule.bSpline2dInit(sys.argv)

		# Read starting model
		modelStartFile=parObject.getString("modelStart","None")
		if (modelStartFile=="None"):
			modelStart=model
			modelStart.scale(0.0)
		else:
			modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=2)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=1)

		# Spline instanciation
		invOp=interpBSpline2dModule.bSpline2d(model,data,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)

	else:
		raise TypeError("ERROR! Please provide an interpolation method")

	############################## Problem #####################################
	# Problem
	invProb=Prblm.ProblemL2Linear(modelStart,data,invOp)

	############################## Solver ######################################
	# Stopper
	stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

	# Folder
	folder=parObject.getString("folder")
	if (os.path.isdir(folder)==False): os.mkdir(folder)
	prefix=parObject.getString("prefix","None")
	if (prefix=="None"): prefix=folder
	invPrefix=folder+"/"+prefix
	logFile=invPrefix+"_logFile"

	# Solver
	LCGsolver=LCG.LCGsolver(stop,logger=logger(logFile))
	LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

	# Run solver
	LCGsolver.run(invProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

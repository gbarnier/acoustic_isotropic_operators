#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import interpBSpline1dModule
import interpRbf1dModule
import numpy as np
import time
import sys

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
		model,data,zOrder,zSplineMesh,zDataAxis,nzParam,scaling,zTolerance,fat=interpBSpline1dModule.bSpline1dInit(sys.argv)

		# Read starting model
		modelStartFile=parObject.getString("modelStart","None")
		if (modelStartFile=="None"):
			modelStart=model
			modelStart.scale(0.0)
		else:
			modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=1)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=1)

		# Spline instanciation
		invOp=interpBSpline1dModule.bSpline1d(modelStart,data,zOrder,zSplineMesh,zDataAxis,nzParam,scaling,zTolerance,fat)

	elif (interp=="rbf"):

		# Rbf initialization
		model,data,epsilon,zSplineMesh,zDataAxis,scaling,fat=interpRbf1dModule.interpRbf1dInit(sys.argv)

		# Read starting model
		modelStartFile=parObject.getString("modelStart","None")
		if (modelStartFile=="None"):
			modelStart=model
			modelStart.scale(0.0)
		else:
			modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=1)

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile,ndims=1)

		# Construct operator
		invOp=interpRbf1dModule.interpRbf1d(model,data,epsilon,zSplineMesh,zDataAxis,scaling,fat)

	else:

		raise TypeError("ERROR! Please provide an interpolation method")

	############################## Problem #####################################
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

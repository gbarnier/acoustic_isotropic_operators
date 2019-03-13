#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import interpBSplineModule
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
	nDim=parObject.getInt("nDim")

	# 1d spline
	if (nDim==1):

		# Initialize 1d spline
		model,data,zOrder,zSplineMesh,zDataAxis,nzParam,scaling,zTolerance,fat=interpBSplineModule.bSpline1dInit(sys.argv)

		# Construct operator
		splineOp=interpBSplineModule.bSpline1d(model,data,zOrder,zSplineMesh,zDataAxis,nzParam,scaling,zTolerance,fat)

	# 2d spline
	if (nDim==2):

		# Initialize 2d spline
		model,data,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSplineModule.bSpline2dInit(sys.argv)

		# Construc 2d spline operator
		splineOp=interpBSplineModule.bSpline2d(model,data,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)

	# 3d spline
	if (nDim==3):

		# Initialize operator
		model,data,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat=interpBSplineModule.bSpline3dInit(sys.argv)

		# Construct operator
		splineOp=interpBSplineModule.bSpline3d(model,data,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat)

	# Read starting model
	modelStartFile=parObject.getString("modelStart","None")
	if (modelStartFile=="None"):
		modelStart=model
		modelStart.scale(0.0)
	else:
		modelStart=genericIO.defaultIO.getVector(modelStartFile,ndims=nDim)

	# Read data
	dataFile=parObject.getString("data")
	data=genericIO.defaultIO.getVector(dataFile,ndims=nDim)

	############################## Problem #####################################
	# Problem
	invProb=Prblm.ProblemL2Linear(modelStart,data,splineOp)

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

	# Solver recording parameters
	iterSampling=parObject.getInt("iterSampling",1)
	bufferSize=parObject.getInt("bufferSize",-1)
	if (bufferSize<0): bufferSize=None

	# Solver
	LCGsolver=LCG.LCGsolver(stop,logger=logger(logFile))
	LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling)

	# Run solver
	LCGsolver.run(invProb,verbose=True)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

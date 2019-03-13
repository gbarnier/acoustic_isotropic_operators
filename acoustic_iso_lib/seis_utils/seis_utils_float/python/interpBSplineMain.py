#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import interpBSplineModule
import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	nDim=parObject.getInt("nDim")
	adj=parObject.getInt("adj",0)
	param=parObject.getInt("meshOut",0) # Set to 1 if you want to write the mesh vectors and other interpolation parameters

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
		startTime=time.time()
		splineOp=interpBSplineModule.bSpline3d(model,data,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat)
		print("Time for constructor = ",time.time()-startTime)

	if (adj==0):

		print("-------------------------------------------------------------------")
		print("-------------- Running Spline interpolation forward ---------------")
		print("---------------------",nDim,"D B-Splines functions -----------------------")
		print("-------------------------------------------------------------------\n")

		# Read model (on coarse grid)
		modelFile=parObject.getString("model")
		model=genericIO.defaultIO.getVector(modelFile)

		# Apply forward
		startTime=time.time()
		splineOp.forward(False,model,data)
		print("Time for forward = ",time.time()-startTime)
		# Write data
		dataFile=parObject.getString("data")
		genericIO.defaultIO.writeVector(dataFile,data)

	else:

		print("-------------------------------------------------------------------")
		print("-------------- Running Spline interpolation adjoint ---------------")
		print("--------------------",nDim,"D B-Splines functions -----------------------")
		print("-------------------------------------------------------------------\n")

		# Read data (fine grid)
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile)

		# Apply adjoint
		startTime=time.time()
		splineOp.adjoint(False,model,data)
		print("Time for adjoint = ",time.time()-startTime)

		# Write model
		modelFile=parObject.getString("model")
		genericIO.defaultIO.writeVector(modelFile,model)

	if (param==1):

		# 1d spline
		if (nDim>0):

			# Write zMeshVector
			zMeshModel=splineOp.getZMeshModel()
			zMeshModelFile=parObject.getString("zMeshModel","junk")
			genericIO.defaultIO.writeVector(zMeshModelFile,zMeshModel)

			# Write zMeshDataVector (fine grid)
			zMeshData=splineOp.getZMeshData()
			zMeshDataFile=parObject.getString("zMeshData","junk")
			genericIO.defaultIO.writeVector(zMeshDataFile,zMeshData)

			# Write control points positions
			zMeshModel1d=splineOp.getZMeshModel1d()
			zMeshModel1dFile=parObject.getString("zMeshModel1d","junk")
			genericIO.defaultIO.writeVector(zMeshModel1dFile,zMeshModel1d)

		# 2d spline
		if (nDim>1):

			# Write zMeshVector
			xMeshModel=splineOp.getXMeshModel()
			xMeshModelFile=parObject.getString("xMeshModel","junk")
			genericIO.defaultIO.writeVector(xMeshModelFile,xMeshModel)

			# Write zMeshDataVector (fine grid)
			xMeshData=splineOp.getXMeshData()
			xMeshDataFile=parObject.getString("xMeshData","junk")
			genericIO.defaultIO.writeVector(xMeshDataFile,xMeshData)

			# Write control points positions
			xMeshModel1d=splineOp.getXMeshModel1d()
			xMeshModel1dFile=parObject.getString("xMeshModel1d","junk")
			genericIO.defaultIO.writeVector(xMeshModel1dFile,xMeshModel1d)

		# 3d spline
		if (nDim==3):

			# Write zMeshVector
			yMeshModel=splineOp.getYMeshModel()
			yMeshModelFile=parObject.getString("yMeshModel","junk")
			genericIO.defaultIO.writeVector(yMeshModelFile,yMeshModel)

			# Write zMeshDataVector (fine grid)
			yMeshData=splineOp.getYMeshData()
			yMeshDataFile=parObject.getString("yMeshData","junk")
			genericIO.defaultIO.writeVector(yMeshDataFile,yMeshData)

			# Write control points positions
			yMeshModel1d=splineOp.getYMeshModel1d()
			yMeshModel1dFile=parObject.getString("yMeshModel1d","junk")
			genericIO.defaultIO.writeVector(yMeshModel1dFile,yMeshModel1d)

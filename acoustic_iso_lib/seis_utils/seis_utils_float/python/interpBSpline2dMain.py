#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import interpBSpline2dModule
import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Initialize operator
	model,data,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dModule.bSpline2dInit(sys.argv)
	# Construct operator
	splineOp=interpBSpline2dModule.bSpline2d(model,data,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("-------------- Running Spline interpolation forward ---------------")
		print("-------------------- 2D B-Splines functions -----------------------")
		print("-------------------------------------------------------------------\n")

		# Read model (on coarse grid)
		modelFile=parObject.getString("model")
		model=genericIO.defaultIO.getVector(modelFile)

		# Apply forward
		splineOp.forward(False,model,data)

		# Write data
		dataFile=parObject.getString("data")
		genericIO.defaultIO.writeVector(dataFile,data)

	else:

		print("-------------------------------------------------------------------")
		print("-------------- Running Spline interpolation adjoint ---------------")
		print("-------------------- 2D B-Splines functions -----------------------")
		print("-------------------------------------------------------------------\n")

		# Read data (fine grid)
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile)

		# Apply adjoint
		splineOp.adjoint(False,model,data)

		# Write model
		modelFile=parObject.getString("model")
		genericIO.defaultIO.writeVector(modelFile,model)


	# Write other interpolation parameters
	zParam=splineOp.getZParamVector()
	xParam=splineOp.getXParamVector()
	zMeshVector=splineOp.getZMesh()
	xMeshVector=splineOp.getXMesh()

	# Write zParam
	zParamFile=parObject.getString("zParam","junk")
	genericIO.defaultIO.writeVector(zParamFile,zParam)

	# Write xParam
	xParamFile=parObject.getString("xParam","junk")
	genericIO.defaultIO.writeVector(xParamFile,xParam)

	# Write zMesh
	zMeshFileOut=parObject.getString("zMeshOut","junk")
	genericIO.defaultIO.writeVector(zMeshFileOut,zSplineMesh)

	# Write xMesh
	xMeshFileOut=parObject.getString("xMeshOut","junk")
	genericIO.defaultIO.writeVector(xMeshFileOut,xSplineMesh)

	# Write zMeshVector
	zMeshVectorFile=parObject.getString("zMeshVector","junk")
	genericIO.defaultIO.writeVector(zMeshVectorFile,zMeshVector)

	# Write xMeshVector
	xMeshVectorFile=parObject.getString("xMeshVector","junk")
	genericIO.defaultIO.writeVector(xMeshVectorFile,xMeshVector)

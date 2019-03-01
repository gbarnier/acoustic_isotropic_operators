#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import interpBSpline1dModule
import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Initialize operator
	model,data,zOrder,zSplineMesh,zDataAxis,nzParam,scaling,zTolerance,fat=interpBSpline1dModule.bSpline1dInit(sys.argv)

	# Construct operator
	splineOp=interpBSpline1dModule.bSpline1d(model,data,zOrder,zSplineMesh,zDataAxis,nzParam,scaling,zTolerance,fat)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("-------------- Running Spline interpolation forward ---------------")
		print("-------------------- 1D B-Splines functions -----------------------")
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
		print("-------------------- 1D B-Splines functions -----------------------")
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
	zMeshVector=splineOp.getZMesh()
	n=zMeshVector.getHyper().axes[0].n

	# Write zMesh
	zMeshFileOut=parObject.getString("zMeshOut","junk")
	genericIO.defaultIO.writeVector(zMeshFileOut,zSplineMesh)

	# Write zMeshVector
	zMeshVectorFile=parObject.getString("zMeshVector","junk")
	genericIO.defaultIO.writeVector(zMeshVectorFile,zMeshVector)

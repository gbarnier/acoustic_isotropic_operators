#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import interpRbf1dModule
import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Initialize operator
	model,data,epsilon,zSplineMesh,zDataAxis,scaling,fat=interpRbf1dModule.interpRbf1dInit(sys.argv)

	# Construct operator
	rbfOp=interpRbf1dModule.interpRbf1d(model,data,epsilon,zSplineMesh,zDataAxis,scaling,fat)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("-------------- Running 1D RBF interpolation forward ---------------")
		print("-------------------------------------------------------------------\n")

		# Read model (on coarse grid)
		modelFile=parObject.getString("model")
		model=genericIO.defaultIO.getVector(modelFile)

		# Apply forward
		rbfOp.forward(False,model,data)

		# Write data
		dataFile=parObject.getString("data")
		genericIO.defaultIO.writeVector(dataFile,data)

	else:

		print("-------------------------------------------------------------------")
		print("-------------- Running 1D RBF interpolation adjoint ---------------")
		print("-------------------------------------------------------------------\n")


		# Read data (fine grid)
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile)

		# Apply adjoint
		rbfOp.adjoint(False,model,data)

		# Write model
		modelFile=parObject.getString("model")
		genericIO.defaultIO.writeVector(modelFile,model)


	# Write other interpolation parameters
	zMeshVector=rbfOp.getZMesh()
	n=zMeshVector.getHyper().axes[0].n
	print("o=",zMeshVector.getNdArray()[0])
	print("f=",zMeshVector.getNdArray()[n-1])

	# Write zMesh
	zMeshFileOut=parObject.getString("zMeshOut","junk")
	genericIO.defaultIO.writeVector(zMeshFileOut,zSplineMesh)

	# Write zMeshVector
	zMeshVectorFile=parObject.getString("zMeshVector","junk")
	genericIO.defaultIO.writeVector(zMeshVectorFile,zMeshVector)

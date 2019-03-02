#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import spatialDerivModule
import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	deriv=parObject.getString("deriv")
	adj=parObject.getInt("adj",0)

	# Forward
	if (adj==0):

		# Read model (on coarse grid)
		modelFile=parObject.getString("model")
		model=genericIO.defaultIO.getVector(modelFile)
		data=model.clone()

		if (deriv=="zGrad"):

			print("-------------- Applying forward gradient in z-direction -------------------")
			fat=spatialDerivModule.zGradInit(sys.argv)
			gradOp=spatialDerivModule.zGradPython(model,data,fat)
			gradOp.forward(False,model,data)

		elif (deriv=="xGrad"):

			print("-------------- Applying forward gradient in x-direction -------------------")
			fat=spatialDerivModule.xGradInit(sys.argv)
			gradOp=spatialDerivModule.xGradPython(model,data,fat)
			gradOp.forward(False,model,data)

		elif (deriv=="Laplacian"):

			print("-------------- Applying forward Laplacian -------------------")
			fat=spatialDerivModule.LaplacianInit(sys.argv)
			gradOp=spatialDerivModule.LaplacianPython(model,data,fat)
			gradOp.forward(False,model,data)

		else:

			print("-------------- Applying forward gradient in zx-direction ------------------")
			fat=spatialDerivModule.zxGradInit(sys.argv)
			gradOp=spatialDerivModule.zxGradPython(model,data,fat)
			gradOp.forward(False,model,data)


		# Write data
		dataFile=parObject.getString("data")
		genericIO.defaultIO.writeVector(dataFile,data)


	# Adjoint
	else:

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile)
		model=data.clone()

		if (deriv=="zGrad"):

			print("-------------- Applying adjoint gradient in z-direction -------------------")
			fat=spatialDerivModule.zGradPython(sys.argv)
			gradOp=spatialDerivModule.zGrad(model,data,fat)
			gradOp.adjoint(False,model,data)

		elif (deriv=="xGrad"):

			print("-------------- Applying adjoint gradient in x-direction -------------------")
			fat=spatialDerivModule.xGradInit(sys.argv)
			gradOp=spatialDerivModule.xGradPython(model,data,fat)
			gradOp.adjoint(False,model,data)

		else:

			print("-------------- Applying adjoint gradient in z-direction -------------------")
			fat=spatialDerivModule.zxGradInit(sys.argv)
			gradOp=spatialDerivModule.zxGradPython(model,data,fat)
			gradOp.adjoint(False,model,data)

		# Write model
		modelFile=parObject.getString("model")
		genericIO.defaultIO.writeVector(modelFile,model)

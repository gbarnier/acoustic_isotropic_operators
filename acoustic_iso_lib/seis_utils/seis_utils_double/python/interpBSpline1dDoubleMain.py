#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import interpBSpline1dDoubleModule
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Initialize spline operator
	model1Double,data1Double,order,splineMesh,dataAxis,nParam,scaling,tolerance,fat=interpBSpline1dDoubleModule.bSpline1dDoubleInit(sys.argv)

	# Create operator
	splineOp=interpBSpline1dDoubleModule.bSpline1dDouble(model1Double,data1Double,order,splineMesh,dataAxis,nParam,scaling,tolerance,fat)

	# Display mesh
	n=splineMesh.getHyper().axes[0].n
	splineMeshNp=splineMesh.getNdArray()
	print("n = ",n)
	for iPoint in range(n):
		# print("i=",iPoint)
		print("splineMesh=",splineMeshNp[iPoint])

	# Read data (velocity) file
	dataFile=parObject.getString("vel")
	data1Float=genericIO.defaultIO.getVector(dataFile)
	data1FloatNp=data1Float.getNdArray()
	data1DoubleNp=data1Double.getNdArray()
	data1DoubleNp[:]=data1FloatNp
	data2Double=data1Double.clone()

	# Apply one adjoint
	splineOp.adjoint(False,model1Double,data1Double)
	splineOp.forward(False,model1Double,data2Double)


	# Write data1
	data1Float=SepVector.getSepVector(data1Double.getHyper(),storage="dataFloat")
	data1FloatNp=data1Float.getNdArray()
	data1FloatNp[:]=data1DoubleNp
	data1File=parObject.getString("data1")
	genericIO.defaultIO.writeVector(data1File,data1Float)

	# Write data2
	data2Float=SepVector.getSepVector(data2Double.getHyper(),storage="dataFloat")
	data2FloatNp=data2Float.getNdArray()
	data2DoubleNp=data2Double.getNdArray()
	data2FloatNp[:]=data2DoubleNp
	data2File=parObject.getString("data2")
	genericIO.defaultIO.writeVector(data2File,data2Float)

	# Write mesh
	meshFloat=SepVector.getSepVector(splineMesh.getHyper(),storage="dataFloat")
	meshFloatNp=meshFloat.getNdArray()
	meshFloatNp[:]=splineMeshNp
	meshFileOut=parObject.getString("meshFileOut")
	genericIO.defaultIO.writeVector(meshFileOut,meshFloat)

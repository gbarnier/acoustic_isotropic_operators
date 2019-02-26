#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import interpBSpline2dDoubleModule
import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	init=parObject.getInt("init")


	if (init==1):

		# Initialize operator and compute parameter vectors
		model1Double,data1Double,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dDoubleModule.bSpline2dDoubleInit(sys.argv)
		splineOp=interpBSpline2dDoubleModule.bSpline2dDouble(model1Double,data1Double,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat,init)

		# Get parameter vectors
		zParam=splineOp.getZParamVector()
		xParam=splineOp.getXParamVector()

		# Write zParam
		zParamNp=zParam.getNdArray()
		zParamFloat=SepVector.getSepVector(zParam.getHyper(),storage="dataFloat")
		zParamFloatNp=zParamFloat.getNdArray()
		zParamFloatNp[:]=zParamNp
		zParamFile=parObject.getString("zParam")
		genericIO.defaultIO.writeVector(zParamFile,zParamFloat)

		# Write xParam
		xParamNp=xParam.getNdArray()
		xParamFloat=SepVector.getSepVector(xParam.getHyper(),storage="dataFloat")
		xParamFloatNp=xParamFloat.getNdArray()
		xParamFloatNp[:]=xParamNp
		xParamFile=parObject.getString("xParam")
		genericIO.defaultIO.writeVector(xParamFile,xParamFloat)

		# Write zMesh
		zSplineMeshNp=zSplineMesh.getNdArray()
		zSplineMeshFloat=SepVector.getSepVector(zSplineMesh.getHyper(),storage="dataFloat")
		zSplineMeshFloatNp=zSplineMeshFloat.getNdArray()
		zSplineMeshFloatNp[:]=zSplineMeshNp
		zMeshFileOut=parObject.getString("zMeshOut")
		genericIO.defaultIO.writeVector(zMeshFileOut,zSplineMeshFloat)

		# Write xMesh
		xSplineMeshNp=xSplineMesh.getNdArray()
		xSplineMeshFloat=SepVector.getSepVector(xSplineMesh.getHyper(),storage="dataFloat")
		xSplineMeshFloatNp=xSplineMeshFloat.getNdArray()
		xSplineMeshFloatNp[:]=xSplineMeshNp
		xMeshFileOut=parObject.getString("xMeshOut")
		genericIO.defaultIO.writeVector(xMeshFileOut,xSplineMeshFloat)

		for i in range(zSplineMeshNp.size):
			print("mesh[",i,"] = ",zSplineMeshNp[i])


	else:

		# Read parameter vectors from command line
		zParamFile=parObject.getString("zParam")
		zParamFloat=genericIO.defaultIO.getVector(zParamFile)
		zParamFloatNp=zParamFloat.getNdArray()
		zParamDouble=SepVector.getSepVector(zParamFloat.getHyper(),storage="dataDouble")
		zParamDoubleNp=zParamDouble.getNdArray()
		zParamDoubleNp[:]=zParamFloatNp
		xParamFile=parObject.getString("xParam")
		xParamFloat=genericIO.defaultIO.getVector(xParamFile)
		xParamFloatNp=xParamFloat.getNdArray()
		xParamDouble=SepVector.getSepVector(xParamFloat.getHyper(),storage="dataDouble")
		xParamDoubleNp=xParamDouble.getNdArray()
		xParamDoubleNp[:]=xParamFloatNp

		# Construct interpolation object
		print("Initializing")
		model1Double,data1Double,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dDoubleModule.bSpline2dDoubleInit(sys.argv)
		print("Creating object")
		splineOp=interpBSpline2dDoubleModule.bSpline2dDouble(model1Double,data1Double,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,zParamDouble,xParamDouble,scaling,zTolerance,xTolerance,fat,init)
		print("Done creating object")

		# Read data (velocity) file
		dataFile=parObject.getString("vel")
		data1Float=genericIO.defaultIO.getVector(dataFile)
		data1FloatNp=data1Float.getNdArray()
		data1DoubleNp=data1Double.getNdArray()
		data1DoubleNp[:]=data1FloatNp
		data2Double=data1Double.clone()

		# Apply one adjoint
		t0=time.time()
		splineOp.adjoint(False,model1Double,data1Double)
		t1=time.time()
		print("Adjoint time=",t1-t0)
		t0=time.time()
		splineOp.forward(False,model1Double,data2Double)
		t1=time.time()
		print("Forward time=",t1-t0)

		# Write model1
		model1Float=SepVector.getSepVector(model1Double.getHyper(),storage="dataFloat")
		model1FloatNp=model1Float.getNdArray()
		model1DoubleNp=model1Double.getNdArray()
		model1FloatNp[:]=model1DoubleNp
		model1File=parObject.getString("model1")
		genericIO.defaultIO.writeVector(model1File,model1Float)

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

		# Write zMesh
		zSplineMeshNp=zSplineMesh.getNdArray()
		zSplineMeshFloat=SepVector.getSepVector(zSplineMesh.getHyper(),storage="dataFloat")
		zSplineMeshFloatNp=zSplineMeshFloat.getNdArray()
		zSplineMeshFloatNp[:]=zSplineMeshNp
		zMeshFileOut=parObject.getString("zMeshOut")
		genericIO.defaultIO.writeVector(zMeshFileOut,zSplineMeshFloat)

		# Write xMesh
		xSplineMeshNp=xSplineMesh.getNdArray()
		xSplineMeshFloat=SepVector.getSepVector(xSplineMesh.getHyper(),storage="dataFloat")
		xSplineMeshFloatNp=xSplineMeshFloat.getNdArray()
		xSplineMeshFloatNp[:]=xSplineMeshNp
		xMeshFileOut=parObject.getString("xMeshOut")
		genericIO.defaultIO.writeVector(xMeshFileOut,xSplineMeshFloat)

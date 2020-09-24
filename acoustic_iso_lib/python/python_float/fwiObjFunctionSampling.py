#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import pyOperator as pyOp
import Acoustic_iso_float
import numpy as np
import time
import sys
from numpy import linalg as LA

#Dask-related modules
import pyDaskOperator as DaskOp

if __name__ == '__main__':

	# Getting parameter object
	parObject=genericIO.io(params=sys.argv)

	# Checking if Dask was requested
	client, nWrks = Acoustic_iso_float.create_client(parObject)

	# Initialize operator
	modelFloat,dataFloat,velFloat,parObject1,sourcesVector,receiversVector,modelFloatLocal=Acoustic_iso_float.nonlinearOpInitFloat(sys.argv,client)

	# Construct nonlinear operator object
	nonlinearOp=Acoustic_iso_float.nonlinearPropShotsGpu(modelFloat,dataFloat,velFloat,parObject1,sourcesVector,receiversVector)

	# Read model
	modelFile=parObject.getString("model","noModelFile")
	modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Read true data
	dataFile=parObject.getString("data","noDataFile")
	dataTrue=genericIO.defaultIO.getVector(dataFile,ndims=3)
	dataTrueNd=dataTrue.getNdArray()
	dataFloatNd=dataFloat.getNdArray()
	res=dataFloat.clone()
	resNd=res.getNdArray()

	# Start looping over velocity values
	velValues = np.linspace(2.0, 3.0, 21)
	velTest=velFloat.clone()
	velTestNd=velTest.getNdArray()
	oVel=velValues[0]
	dVel=velValues[1]-velValues[0]
	nVel=len(velValues)

	# Get model dimensions
	nz=parObject.getInt("nz")
	nx=parObject.getInt("nx")
	fat=parObject.getInt("fat")

	# Create objective function value array
	velAxis=Hypercube.axis(n=nVel,o=oVel,d=dVel)
	objFuncHyper=Hypercube.hypercube(axes=[velAxis])
	objFunc=SepVector.getSepVector(objFuncHyper)
	# objFunc.set(2.0)
	objFuncNd=objFunc.getNdArray()

	# Loop over velocity values
	for iVel in range(len(velValues)):

		# print("iVel = ",iVel)
		# print("vel = ",velValues[iVel])
		# print("dataTrue norm = ",dataTrue.norm())

		# Fill velocity model with current velocity value
		for ix in range(fat,nx-fat):
			for iz in range(fat,nz-fat):
				velTestNd[ix][iz]=velValues[iVel]

		# print("velTest max = ",velTest.max())
		# print("velTest min = ",velTest.min())

		# Reste velocity in nonlinear operator
		nonlinearOp.setVel(velTest)

		# Launch forward
		nonlinearOp.forward(False,modelFloat,dataFloat)
		# print("dataFloat norm = ",dataFloat.norm())

		# Compute FWI objective function value
		# for iShot in range(dataFloat.getHyper().axes[2].n):
		# 	for iRec in range(dataFloat.getHyper().axes[1].n):
		# 		for iTime in range(dataFloat.getHyper().axes[0].n):
		# 			resNd[iShot][iRec][iTime]=dataFloatNd[iShot][iRec][iTime]-dataTrueNd[iShot][iRec][iTime]

		resNd[:][:][:]=dataFloatNd[:][:][:]-dataTrueNd[:][:][:]

		objFuncNd[iVel]=0.5*res.norm()*res.norm()
		# print("objFuncNd[iVel] = ",objFuncNd[iVel])

	# Write data
	objFuncFile=parObject.getString("objFunction")
	objFunc.writeVec(objFuncFile)

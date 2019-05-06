#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import numpy as np
import time
import sys

if __name__ == '__main__':

	# Initialize operator
	model,seismicData,velFloat,parObject,sourcesVector,receiversVector=Acoustic_iso_float.nonlinearOpInitFloat(sys.argv)

	# Construct nonlinear operator object
	nonlinearOp=Acoustic_iso_float.nonlinearPropShotsGpu(model,seismicData,velFloat,parObject,sourcesVector,receiversVector)

	print("-------------------------------------------------------------------")
	print("------------- Generating Fwi objective function from FWIME --------")
	print("-------------------- Single precision Python code -----------------")
	print("-------------------------------------------------------------------\n")

	# Check that model was provided
	modelFile=parObject.getString("model","noModelFile")
	if (modelFile == "noModelFile"):
		print("**** ERROR: User did not provide model file ****\n")
		quit()

	# Read sources
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Read velocity models
	velocityFwimeFile=parObject.getString("velocityFwime")
	velocityFwime=genericIO.defaultIO.getVector(velocityFwimeFile,ndims=3)
	velocityFwimeNd=velocityFwime.getNdArray()

	# Get number of inverted models
	nIter=velocityFwime.getHyper().axes[2].n

	# Generate a 2d velocity model
	zAxis=velocityFwime.getHyper().axes[0]
	xAxis=velocityFwime.getHyper().axes[1]
	velTemp=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis]))
	velTempNd=velTemp.getNdArray()

	# Read observed data
	obsDataFile=parObject.getString("obsData")
	obsData=genericIO.defaultIO.getVector(obsDataFile,ndims=3)
	# obsDataNd=obsData.getNdArray()

	# Allocate Fwi objective function
	iterAxis=velocityFwime.getHyper().axes[2]
	objFunction=SepVector.getSepVector(Hypercube.hypercube(axes=[iterAxis]))
	objFunctionNd=objFunction.getNdArray()

	for iIter in range(nIter):

		print("iter #",iIter)

		# Read the new velocity model
		velTempNd[:] = velocityFwimeNd[iIter,:,:]

		# Set the velocity
		nonlinearOp.setVel(velTemp)

		# Apply forward
		nonlinearOp.forward(False,model,seismicData)

		# Compute data difference
		seismicData.scaleAdd(obsData,1,-1)

		# Compute Fwi objective function
		objFunctionNd[iIter]=0.5*seismicData.norm()*seismicData.norm()

	# Write objective function value
	dataFile=parObject.getString("data","noDataFile")
	genericIO.defaultIO.writeVector(dataFile,objFunction)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

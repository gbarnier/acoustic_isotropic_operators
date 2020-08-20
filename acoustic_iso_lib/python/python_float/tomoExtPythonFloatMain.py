#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import pyOperator as pyOp
import Acoustic_iso_float
import numpy as np
import time
import sys

#Dask-related modules
import pyDaskOperator as DaskOp

if __name__ == '__main__':

	#Getting parameter object
	parObject=genericIO.io(params=sys.argv)

	# Checking if Dask was requested
	client, nWrks = Acoustic_iso_float.create_client(parObject)

	# Seismic operator object initialization
	modelFloat,dataFloat,velFloat,parObject1,sourcesVector,sourcesSignalsVector,receiversVector,reflectivityFloat,modelFloatLocal=Acoustic_iso_float.tomoExtOpInitFloat(sys.argv,client)

	if client:
		#Instantiating Dask Operator
		tomoExtOp_args = [(modelFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsVector[iwrk],receiversVector[iwrk],reflectivityFloat.vecDask[iwrk]) for iwrk in range(nWrks)]
		tomoExtOp = DaskOp.DaskOperator(client,Acoustic_iso_float.tomoExtShotsGpu,tomoExtOp_args,[1]*nWrks)
		#Adding spreading operator and concatenating with Born operator (using modelFloatLocal)
		Sprd = DaskOp.DaskSpreadOp(client,modelFloatLocal,[1]*nWrks)
		tomoExtOp = pyOp.ChainOperator(Sprd,tomoExtOp)
	else:
		# Construct Tomo operator object
		tomoExtOp=Acoustic_iso_float.tomoExtShotsGpu(modelFloat,dataFloat,velFloat,parObject1,sourcesVector,sourcesSignalsVector,receiversVector,reflectivityFloat)

	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		tomoExtOp.dotTest(True)
		quit(0)

	# Forward
	if (parObject.getInt("adj", 0) == 0):

		print("-------------------------------------------------------------------")
		print("--------------- Running Python tomo extended forward --------------")
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			print("**** ERROR: User did not provide model file ****\n")
			quit()
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			print("**** ERROR: User did not provide data file name ****\n")
			quit()

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=2)

		# Apply forward
		tomoExtOp.forward(False,modelFloat,dataFloat)

		# Write data
		dataFloat.writeVec(dataFile)

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("---------------- Running Python tomo extended adjoint -------------")
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		# Check that data was provided
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			print("**** ERROR: User did not provide data file ****\n")
			quit()
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			print("**** ERROR: User did not provide model file name ****\n")
			quit()

		# Read data
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Apply adjoint
		tomoExtOp.adjoint(False,modelFloatLocal,dataFloat)

		# Write model
		modelFloatLocal.writeVec(modelFile)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

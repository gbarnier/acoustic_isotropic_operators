#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import pyOperator as pyOp
import Acoustic_iso_float
import numpy as np
import sys

#Dask-related modules
from dask_util import DaskClient
import pyDaskOperator as DaskOp

if __name__ == '__main__':

	#Getting parameter object
	parObject=genericIO.io(params=sys.argv)
	hostnames = parObject.getString("hostnames","noHost")
	client = None
	#Starting Dask client if requested
	if(hostnames != "noHost"):
		print("Starting Dask client using the following workers: %s"%(hostnames))
		client = DaskClient(hostnames=hostnames.split(","))
		print("Client has started!")
		nWrks = client.getNworkers()

	# Seismic operator object initialization
	modelFloat,dataFloat,velFloat,parObject1,sourcesVector,sourcesSignalsVector,receiversVector,modelFloatLocal=Acoustic_iso_float.BornExtOpInitFloat(sys.argv,client)

	if client:
		#Instantiating Dask Operator
		BornExtOp_args = [(modelFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat[iwrk],parObject1[iwrk],sourcesVector[iwrk],sourcesSignalsVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		BornExtOp = DaskOp.DaskOperator(client,Acoustic_iso_float.BornExtShotsGpu,BornExtOp_args,[1]*nWrks)
		#Adding spreading operator and concatenating with Born operator (using modelFloatLocal)
		Sprd = DaskOp.DaskSpreadOp(client,modelFloatLocal,[1]*nWrks)
		BornExtOp = pyOp.ChainOperator(Sprd,BornExtOp)

	else:
		# Construct Born operator object
		BornExtOp=Acoustic_iso_float.BornExtShotsGpu(modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		BornExtOp.dotTest(True)
		quit(0)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("--------------- Running Python Born extended forward --------------")
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			print("**** ERROR: User did not provide model file ****\n")
			quit()

		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file name ****\n")

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)

		# Apply forward
		BornExtOp.forward(False,modelFloat,dataFloat)

		# Write data
		dataFloat.writeVec(dataFile)

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("---------------- Running Python extended Born adjoint -------------")
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
		if(client):
			#Chunking the data and spreading them across workers if dask was requested
			dataFloat = Acoustic_iso_float.chunkData(dataFloat,BornExtOp.getRange())

		# Apply adjoint
		BornExtOp.adjoint(False,modelFloatLocal,dataFloat)

		# Write model
		modelFloatLocal.writeVec(modelFile)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

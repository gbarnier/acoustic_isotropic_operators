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

	# Initialize operator
	modelFloat,dataFloat,velFloat,parObject1,sourcesVector,receiversVector,modelFloatLocal=Acoustic_iso_float.nonlinearOpInitFloat(sys.argv,client)

	if(client):
		#Instantiating Dask Operator
		nlOp_args = [(modelFloat.vecDask[iwrk],dataFloat.vecDask[iwrk],velFloat[iwrk],parObject1[iwrk],sourcesVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		nonlinearOp = DaskOp.DaskOperator(client,Acoustic_iso_float.nonlinearPropShotsGpu,nlOp_args,[1]*nWrks)
		#Adding spreading operator and concatenating with non-linear operator (using modelFloatLocal)
		Sprd = DaskOp.DaskSpreadOp(client,modelFloatLocal,[1]*nWrks)
		nonlinearOp = pyOp.ChainOperator(Sprd,nonlinearOp)
	else:
		# Construct nonlinear operator object
		nonlinearOp=Acoustic_iso_float.nonlinearPropShotsGpu(modelFloat,dataFloat,velFloat,parObject1,sourcesVector,receiversVector)

	#Testing dot-product test of the operator
	if (parObject.getInt("dpTest",0) == 1):
		nonlinearOp.dotTest(True)
		quit(0)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("------------------ Running Python nonlinear forward ---------------")
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise IOError("**** ERROR: User did not provide model file ****\n")

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)

		# Write data
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file name ****\n")

		if (parObject.getInt("saveWavefield",0) == 1):
			wfldFile=parObject.getString("wfldFile","noWfldFile")
			if (wfldFile == "noWfldFile"):
				raise IOError("**** ERROR: User specified saveWavefield=1 but did not provide wavefield file name (wfldFile)****")

			# Run Nonlinear forward with wavefield saving
			nonlinearOp.forwardWavefield(False,modelFloat,dataFloat)

			# Save wavefield to disk
			wavefieldFloat = nonlinearOp.getWfld()
			wavefieldFloat.writeVec(wfldFile)
		else:
			# Apply forward
			nonlinearOp.forward(False,modelFloat,dataFloat)
		# genericIO.defaultIO.writeVector(dataFile,dataFloat)
		dataFloat.writeVec(dataFile)

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("----------------- Running Python nonlinear adjoint ----------------")
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		# Check that data was provided
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			raise IOError("**** ERROR: User did not provide data file ****\n")

		# Read data
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)
		if(client):
			#Chunking the data and spreading them across workers if dask was requested
			dataFloat = Acoustic_iso_float.chunkData(dataFloat,nonlinearOp.getRange())

		# Apply adjoint
		nonlinearOp.adjoint(False,modelFloatLocal,dataFloat)

		# Write model
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			raise IOError("**** ERROR: User did not provide model file name ****\n")

		modelFloatLocal.writeVec(modelFile)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

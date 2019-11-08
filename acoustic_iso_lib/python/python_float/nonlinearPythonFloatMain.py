#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import pyOperator as pyOp
import Acoustic_iso_float
import numpy as np
import time
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
		print("Starting dask client using following workers: %s"%(hostnames))
		client = DaskClient(hostnames.split(","))
		print("Client has started!")

	# Initialize operator
	modelFloat,dataFloat,velFloat,parObject1,sourcesVector,receiversVector=Acoustic_iso_float.nonlinearOpInitFloat(sys.argv,client)

	if(client):
		quit()
		#Instantiating Dask Operator
		nWrks = client.getNworkers()
		nlOp_args = [(modelFloat[iwrk],dataFloat[iwrk],velFloat[iwrk],parObject1[iwrk],sourcesVector[iwrk],receiversVector[iwrk]) for iwrk in range(nWrks)]
		nonlinearOp = DaskOp.DaskOperator(client,Acoustic_iso_float.nonlinearOpInitFloat,scal_op_args,[1]*nWrks)
	else:
		# Construct nonlinear operator object
		nonlinearOp=Acoustic_iso_float.nonlinearPropShotsGpu(modelFloat,dataFloat,velFloat,parObject1.param,sourcesVector,receiversVector)

	# Forward
	if (parObject.getInt("adj",0) == 0):

		print("-------------------------------------------------------------------")
		print("------------------ Running Python nonlinear forward ---------------")
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		# Check that model was provided
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			print("**** ERROR: User did not provide model file ****\n")
			quit()

		# Read model
		modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)

		#Adding spreading operator and concatenating with non-linear operator
		if(client):
			Sprd = DaskOp.DaskSpreadOp(client,modelFloat,[1]*nWrks)
			nonlinearOp = pyOp.ChainOperator(Sprd,nonlinearOp)

		# Apply forward
		nonlinearOp.forward(False,modelFloat,dataFloat)

		# Write data
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			print("**** ERROR: User did not provide data file name ****\n")
			quit()
		genericIO.defaultIO.writeVector(dataFile,dataFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

	# Adjoint
	else:

		print("-------------------------------------------------------------------")
		print("----------------- Running Python nonlinear adjoint ----------------")
		print("-------------------- Single precision Python code -----------------")
		print("-------------------------------------------------------------------\n")

		# Check that data was provided
		dataFile=parObject.getString("data","noDataFile")
		if (dataFile == "noDataFile"):
			print("**** ERROR: User did not provide data file ****\n")
			quit()

		# Read data
		dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)

		# Apply adjoint
		nonlinearOp.adjoint(False,modelFloat,dataFloat)

		# Write model
		modelFile=parObject.getString("model","noModelFile")
		if (modelFile == "noModelFile"):
			print("**** ERROR: User did not provide model file name ****\n")
			quit()
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

		print("-------------------------------------------------------------------")
		print("--------------------------- All done ------------------------------")
		print("-------------------------------------------------------------------\n")

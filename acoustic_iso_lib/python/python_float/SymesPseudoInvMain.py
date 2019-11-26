#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import numpy as np
import time
import sys
import time

if __name__ == '__main__':

	# Seismic operator object initialization
	modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,dts,fat,taperEndTraceWidth=Acoustic_iso_float.SymesPseudoInvInit(sys.argv)

	# Construct Born operator object
	SymesPseudoInvOp=Acoustic_iso_float.SymesPseudoInvGpu(modelFloat,dataFloat,velFloat,parObject.param,sourcesVector,sourcesSignalsVector,receiversVector,dts,fat,taperEndTraceWidth)

	print("-------------------------------------------------------------------")
	print("------------------- Running Symes' pseudo inverse -----------------")
	print("-------------------- Single precision Python code -----------------")
	print("-------------------------------------------------------------------\n")

	# Check that model was provided
	modelFile=parObject.getString("model","noModelFile")
	if (modelFile == "noModelFile"):
		print("**** ERROR: User did not provide model file ****\n")
		quit()

	# Read model
	modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Apply forward
	SymesPseudoInvOp.forward(False,modelFloat,dataFloat)

	# Write data
	dataFile=parObject.getString("data","noDataFile")
	if (dataFile == "noDataFile"):
		print("**** ERROR: User did not provide data file name ****\n")
		quit()
	genericIO.defaultIO.writeVector(dataFile,dataFloat)

	print("-------------------------------------------------------------------")
	print("--------------------------- All done ------------------------------")
	print("-------------------------------------------------------------------\n")

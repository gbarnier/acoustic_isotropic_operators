#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import dataTaperDoubleModule
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Read model (seismic data that you wihs to mute/taper)
	modelFile=parObject.getString("model")
	modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)
	modelDouble=SepVector.getSepVector(modelFloat.getHyper(),storage="dataDouble")
	modelFloatNp=modelFloat.getNdArray()
	modelDoubleNp=modelDouble.getNdArray()
	modelDoubleNp[:]=modelFloatNp

	# Read parameters
	maxOffset=parObject.getFloat("maxOffset",0) # After that offset, starting tapering and muting [km]
	exp=parObject.getFloat("exp",2) # Coeffiecient that control the steepeness of the tapering
	taperWidth=parObject.getFloat("taperWidth",0) # Tapering width [km]
	muteType=parObject.getString("muteType","offset") # Type of data muting
	maskOnly=parObject.getInt("maskOnly",0)  # If maskOnly=1, only compute and write the mask

	# Allocate data (tapered seismic data) and tapering mask
	if (maskOnly == 0):
		dataDouble=SepVector.getSepVector(modelFloat.getHyper(),storage="dataDouble")

	taperMaskDouble=SepVector.getSepVector(modelFloat.getHyper(),storage="dataDouble")

	# Instanciate dataTaper object
	dataTaperOb=dataTaperDoubleModule.dataTaperDouble(modelDouble,modelDouble,maxOffset,exp,taperWidth,modelFloat.getHyper(),muteType)

	# Get tapering mask numpy array
	taperMaskDoubleNp=np.array(dataTaperOb.getTaperMask(),copy=False)

	# Apply tapering mask
	if (maskOnly == 0):

		# Apply taper to data
		dataTaperOb.forward(False,modelDouble,dataDouble)

		# Write data
		dataFloat=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat")
		dataFloatNp=dataFloat.getNdArray()
		dataDoubleNp=dataDouble.getNdArray()
		dataFloatNp[:]=dataDoubleNp
		dataFile=parObject.getString("data")
		genericIO.defaultIO.writeVector(dataFile,dataFloat)

	# Write taper mask
	taperMaskFloat=SepVector.getSepVector(modelDouble.getHyper(),storage="dataFloat")
	taperMaskFloatNp=taperMaskFloat.getNdArray()
	taperMaskFloatNp[:]=taperMaskDoubleNp
	taperMaskFile=parObject.getString("taperMask")
	genericIO.defaultIO.writeVector(taperMaskFile,taperMaskFloat)

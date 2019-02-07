#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import dataTaperFloatModule
import sys

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Read model (seismic data that you wihs to mute/taper)
	modelFile=parObject.getString("model")
	modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)
	muteType=parObject.getString("muteType")

	if (muteType == "offset"):

		print("Applying offset muting")

		# Read parameters for offset muting
		maxOffset=parObject.getFloat("maxOffset",0) # After that offset, starting tapering and muting [km]
		exp=parObject.getFloat("exp",2) # Coeffiecient that control the steepeness of the tapering
		taperWidth=parObject.getFloat("taperWidth",0) # Tapering width [km]
		maskOnly=parObject.getInt("maskOnly",0)  # If maskOnly=1, only compute and write the mask

		# Instanciate dataTaper object
		dataTaperOb=dataTaperFloatModule.dataTaperFloat(modelFloat,modelFloat,maxOffset,exp,taperWidth,modelFloat.getHyper())

	if (muteType == "time"):

		print("Applying time muting")

		# Read parameters for time muting
		t0=parObject.getFloat("t0",0) # Time origin for time muting
		velMute=parObject.getFloat("velMute",0.0) # Muting velocity
		print("velMute = ",velMute)
		moveout=parObject.getString("moveout","linear") # Time of moveout for the muting
		taperWidth=parObject.getFloat("taperWidth",0) # Tapering width [s]
		exp=parObject.getFloat("exp",2) # Coeffiecient that control the steepeness of the tapering
		maskOnly=parObject.getInt("maskOnly",0)  # If maskOnly = 1, only compute and write the mask

		# Instanciate dataTaper object
		dataTaperOb=dataTaperFloatModule.dataTaperFloat(modelFloat,modelFloat,t0,velMute,exp,taperWidth,modelFloat.getHyper(),moveout)

	# Get tapering mask numpy array
	taperMaskFloat=dataTaperOb.getTaperMask()

	# Write taper mask
	taperMaskFile=parObject.getString("taperMask")
	genericIO.defaultIO.writeVector(taperMaskFile,taperMaskFloat)

	# Apply tapering mask
	if (maskOnly == 0):

		# Allocate data (tapered seismic data) and tapering mask
		dataFloat=SepVector.getSepVector(modelFloat.getHyper())

		# Apply taper to data
		dataTaperOb.forward(False,modelFloat,dataFloat)

		# Write data
		dataFile=parObject.getString("data")
		genericIO.defaultIO.writeVector(dataFile,dataFloat)

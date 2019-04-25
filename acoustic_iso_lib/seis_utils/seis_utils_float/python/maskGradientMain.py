#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import maskGradientModule
import sys

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Read model (seismic data that you wihs to mute/taper)
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile)

	# Initialize operator
	vel,bufferUp,bufferDown,taperExp,fat,wbShift=maskGradientModule.maskGradientInit(sys.argv)

	# Instanciate operator
	maskGradientOp=maskGradientModule.maskGradient(model,model,vel,bufferUp,bufferDown,taperExp,fat,wbShift)

	# Get tapering mask
	maskFile=parObject.getString("mask","noMaskFile")
	if (maskFile != "noMaskFile"):
		taperMask=maskGradientOp.getMask()
		genericIO.defaultIO.writeVector(maskFile,taperMask)

	# Write data
	data=SepVector.getSepVector(model.getHyper())
	maskGradientOp.forward(False,model,data)
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)

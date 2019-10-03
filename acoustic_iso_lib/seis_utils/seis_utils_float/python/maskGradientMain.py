#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import maskGradientModule
import sys

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Initialize operator
	vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile=maskGradientModule.maskGradientInit(sys.argv)

	# Read model
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=2)

	# Instanciate operator
	maskGradientOp=maskGradientModule.maskGradient(vel,vel,vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile)

	# Get tapering mask and write to output
	maskFile=parObject.getString("mask")
	mask=maskGradientOp.getMask()
	genericIO.defaultIO.writeVector(maskFile,mask)

	# Apply forward operator and write output data
	data=SepVector.getSepVector(model.getHyper())
	maskGradientOp.adjoint(False,data,model)
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)

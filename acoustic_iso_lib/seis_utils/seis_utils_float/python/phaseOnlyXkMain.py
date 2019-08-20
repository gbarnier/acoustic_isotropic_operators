#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import phaseOnlyXkModule
import sys
import time
from numpy import linalg as LA

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Read model
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Create data
	data=model.clone()

	# Instanciate PhaseOnlyXk object
	phaseOnlyXkOp=phaseOnlyXkModule.phaseOnlyXk(data,data)

	# Apply normalization
	phaseOnlyXkOp.forward(False,model,data)

	# Write data
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)

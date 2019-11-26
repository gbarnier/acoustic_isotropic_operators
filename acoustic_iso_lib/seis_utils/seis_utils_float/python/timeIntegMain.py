#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import timeIntegModule
import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Read model (seismic data)
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)
	data=model.clone()

	# Initialize and instanciate time integration operator
	dts=timeIntegModule.timeIntegInit(sys.argv)
	timeIntegOp=timeIntegModule.timeInteg(model,dts)

	# Apply forward
	timeIntegOp.forward(False,model,data)

	# Write data
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)

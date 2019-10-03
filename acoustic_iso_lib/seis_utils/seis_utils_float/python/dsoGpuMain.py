#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import dsoGpuModule
import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)
	adj=parObject.getInt("adj",0)

	nz,nx,nExt,fat,zeroShift=dsoGpuModule.dsoGpuInit(sys.argv)

	# Forward
	if (adj==0):

		# Read model (on coarse grid)
		modelFile=parObject.getString("model")
		model=genericIO.defaultIO.getVector(modelFile)

		# Create data
		data=model.clone()

		# Create DSO object and run forward
		dsoOp=dsoGpuModule.dsoGpu(model,data,nz,nx,nExt,fat,zeroShift)
		dsoOp.forward(False,model,data)

		# Write data
		dataFile=parObject.getString("data")
		genericIO.defaultIO.writeVector(dataFile,data)


	# Adjoint
	else:

		# Read data
		dataFile=parObject.getString("data")
		data=genericIO.defaultIO.getVector(dataFile)

		# Create model
		model=data.clone()

		# Create DSO object and run forward
		dsoOp=dsoGpuModule.dsoGpu(model,data,nz,nx,nExt,fat,zeroShift)
		dsoOp.adjoint(False,model,data)

		# Write model
		modelFile=parObject.getString("model")
		genericIO.defaultIO.writeVector(modelFile,model)

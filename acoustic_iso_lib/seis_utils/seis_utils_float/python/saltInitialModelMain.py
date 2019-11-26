#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Read true model
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=2)
	modelNp=model.getNdArray()

	# Time Axis
	# nts=parObject.getInt("nts",-1)
	# ots=parObject.getFloat("ots",0.0)
	# dts=parObject.getFloat("dts",-1.0)
	# timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate model and fill with zeros
	# dummyAxis=Hypercube.axis(n=1)
	# modelHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis,dummyAxis])
	# modelFloat=SepVector.getSepVector(modelHyper)

	############################################################################
	######################### Find water bottom index ##########################
	############################################################################
	# Get dimensions
	nz=model.getHyper().axes[0].n
	oz=model.getHyper().axes[0].o
	dz=model.getHyper().axes[0].d
	nx=model.getHyper().axes[1].n
	modelNp=model.getNdArray()
	xAxis=model.getHyper().axes[1]

	# Data
	data=SepVector.getSepVector(model.getHyper())
	dataNp=data.getNdArray()

	# Create water bottom INDEX array
	wbIndexHyper=Hypercube.hypercube(axes=[xAxis])
	wbIndex=SepVector.getSepVector(wbIndexHyper)
	wbIndexNp=wbIndex.getNdArray()

	# Create water bottom DEPTH array
	wbDepth=SepVector.getSepVector(wbIndex.getHyper())
	wbDepthNp=wbDepth.getNdArray()

	# Get water velocity (assumed to be constant)
	waterVelocity=modelNp[0][0]
	print("Minimum model velocity",np.amin(modelNp))
	print("Maximum model velocity",np.amax(modelNp))
	print("Water velocity",waterVelocity)

	# Compute water bottom index and depth
	for ix in range(nx):
		iz=0
		while (modelNp[ix][iz] == waterVelocity):
			wbIndexNp[ix]=iz
			wbDepthNp[ix]=(iz-1)*dz+oz
			iz=iz+1

	for ix in range(nx):
		for iz in range(nz):
			if (iz < wbIndexNp[ix]):
				dataNp[ix][iz]=1.3
			else:
				dataNp[ix][iz]=1.5

	# write water bottom index
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)

	# write water bottom index
	wbIndexFile=parObject.getString("wbIndex")
	genericIO.defaultIO.writeVector(wbIndexFile,wbIndex)

	# write water bottom index
	wbDepthFile=parObject.getString("wbDepth")
	genericIO.defaultIO.writeVector(wbDepthFile,wbDepth)

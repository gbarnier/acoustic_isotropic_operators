#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import dataTaperModule
import sys

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Read model (seismic data that you wihs to mute/taper)
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Initialize operator
	t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec=dataTaperModule.dataTaperInit(sys.argv)

	# Instanciate operator
	dataTaperOb=dataTaperModule.datTaper(model,model,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,model.getHyper(),time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec)

	# Get tapering mask
	taperMask=dataTaperOb.getTaperMask()
	maskFile=parObject.getString("mask")
	genericIO.defaultIO.writeVector(maskFile,taperMask)

	# Write data
	data=SepVector.getSepVector(model.getHyper())
	dataTaperOb.forward(False,model,data)
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)

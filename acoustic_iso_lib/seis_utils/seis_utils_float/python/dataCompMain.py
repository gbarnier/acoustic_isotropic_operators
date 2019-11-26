#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)
	
	offset=parObject.getString("offset","pos")
	# xShot=parObject.getFloat("xShot") # Shot position [km]
	iShot=parObject.getInt("iShot")
	res=parObject.getInt("res",1) # Flag to determined whether the input model is a predicted data or data residual

	# Read true data
	obsDataFile=parObject.getString("obsIn")
	obsData=genericIO.defaultIO.getVector(obsDataFile,ndims=2)
	obsDataNp=obsData.getNdArray()

	# Read data residuals or predicted data
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)
	modelNp=model.getNdArray()

	# Iteration
	oIter=model.getHyper().axes[2].o
	dIter=model.getHyper().axes[2].d
	nIter=model.getHyper().axes[2].n
	iterAxis=Hypercube.axis(n=nIter,o=oIter,d=dIter,label="Iteration #")

	# Shot
	oShotGrid=parObject.getInt("xSource")
	dShotGrid=parObject.getInt("spacingShots")

	# Receiver
	oRec=obsData.getHyper().axes[1].o
	dRec=obsData.getHyper().axes[1].d
	nRec=obsData.getHyper().axes[1].n
	recAxis=Hypercube.axis(n=nRec,o=oRec,d=dRec,label="Receivers [km]")

	# Time
	ots=obsData.getHyper().axes[0].o
	dts=obsData.getHyper().axes[0].d
	nts=obsData.getHyper().axes[0].n
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts,label="Time [s]")

	# Find indices on the "receiver grid"
	iShotRecGrid=oShotGrid+iShot*dShotGrid
	xShot=oRec+iShotRecGrid*dRec
	print("xShot=",xShot)
	print("iShot=",iShot)
	print("iShotRecGrid=",iShotRecGrid)

	# If user provides residual data instead of predicted data
	if(res==1):
		for iIter in range(nIter):
			modelNp[iIter][:][:]=modelNp[iIter][:][:]+obsDataNp[:][:]

	# Find the number of traces for each side
	if(offset=="pos"):
		nRecNew=nRec-iShotRecGrid-1
	else:
		nRecNew=iShotRecGrid

	# Total number of receivers for the super shot gather
	nRecTotal=2*nRecNew+1

	# Allocate super shot gather
	recNewAxis=Hypercube.axis(n=nRecTotal,o=-nRecNew*dRec,d=dRec,label="Offset [km]")
	iterAxis=Hypercube.axis(n=nIter,o=oIter,d=dIter,label="Iteration #")
	# shotNewAxis=Hypercube.axis(n=1,o=xShotNew,d=1.0)
	superShotGatherHyper=Hypercube.hypercube(axes=[timeAxis,recNewAxis,iterAxis])
	superShotGather=SepVector.getSepVector(superShotGatherHyper)
	superShotGather.scale(0.0)
	superShotGatherNp=superShotGather.getNdArray()

	# Allocate observed data
	# obsHyper=Hypercube.hypercube(axes=[timeAxis,recAxis])
	# obs=SepVector.getSepVector(obsHyper)
	# obsNp=obs.getNdArray()
	# predHyper=Hypercube.hypercube(axes=[timeAxis,recAxis,iterAxis])
	# pred=SepVector.getSepVector(predHyper)
	# predNp=pred.getNdArray()

	# Copy value to super shot gather
	if (offset=="pos"):
		for iIter in range(nIter):
			for iRec in range(nRecNew):
				for its in range(nts):
					superShotGatherNp[iIter][iRec][its]=obsDataNp[iShotRecGrid+nRecNew-iRec][its]
					superShotGatherNp[iIter][iRec+nRecNew+1][its]=modelNp[iIter][iShotRecGrid+iRec+1][its]
			superShotGatherNp[iIter][nRecNew][:]=obsDataNp[iShotRecGrid][:]
			# predNp[iIter][:][:]=modelNp[iIter][:][:]

	else:
		for iIter in range(nIter):
			for iRec in range(nRecNew):
				superShotGatherNp[iIter][iRec][:]=obsDataNp[iRec][:]
				superShotGatherNp[iIter][iRec+nRecNew+1][:]=modelNp[iIter][iShotRecGrid-iRec-1][:]
			superShotGatherNp[iIter][nRecNew][:]=obsDataNp[iShotRecGrid][:]
			# predNp[iIter][:][:]=modelNp[iIter][:][:]

	# obsNp[:][:]=obsDataNp[iShot][:][:]

	# Write super shot gather
	superShotGatherFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(superShotGatherFile,superShotGather)

	# Write observed shot gather
	# obsOutFile=parObject.getString("obsOut")
	# genericIO.defaultIO.writeVector(obsOutFile,obs)
	#
	# # Write predicted shot gather
	# predFile=parObject.getString("pred")
	# genericIO.defaultIO.writeVector(predFile,pred)

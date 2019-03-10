#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	offset=parObject.getString("offset","pos")
	xShot=parObject.getFloat("xShot") # Shot position [km]
	res=parObject.getInt("res",1)

	# Read true data
	obsDataFile=parObject.getString("obs")
	obsData=genericIO.defaultIO.getVector(obsDataFile,ndims=3)
	obsDataNp=obsData.getNdArray()

	# Read data residuals
	modelDataFile=parObject.getString("model")
	modelData=genericIO.defaultIO.getVector(modelDataFile,ndims=3)
	modelDataNp=modelData.getNdArray()
	if(res==1):
		modelData.scaleAdd(obsData,1.0,1.0)

	# Find the shot index for that location
	oShot=obsData.getHyper().axes[2].o
	dShot=obsData.getHyper().axes[2].d
	nShot=obsData.getHyper().axes[2].n
	oShotGrid=parObject.getInt("xSource")
	dShotGrid=parObject.getInt("spacingShots")


	# Receiver
	oRec=obsData.getHyper().axes[1].o
	dRec=obsData.getHyper().axes[1].d
	nRec=obsData.getHyper().axes[1].n

	# Time
	ots=obsData.getHyper().axes[0].o
	dts=obsData.getHyper().axes[0].d
	nts=obsData.getHyper().axes[0].n
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts,label="Time [s]")

	# Find indices on the "shot grid" and "receiver grid"
	iShot=int((xShot-oShot)/dShot)
	iShotRecGrid=oShotGrid+iShot*dShotGrid
	xShotNew=oRec+iShotRecGrid*dRec

	# Find the number of traces for each side
	if(offset=="pos"):
		nRecNew=nRec-iShotRecGrid-1
	else:
		nRecNew=iShotRecGrid

	# Total number of receivers for the super shot gather
	nRecTotal=2*nRecNew+1

	# Allocate super shot gather
	recNewAxis=Hypercube.axis(n=nRecTotal,o=-nRecNew*dRec,d=dRec,label="Offset [km]")
	shotNewAxis=Hypercube.axis(n=1,o=xShotNew,d=1.0)
	superShotGatherHyper=Hypercube.hypercube(axes=[timeAxis,recNewAxis,shotNewAxis])
	superShotGather=SepVector.getSepVector(superShotGatherHyper)
	superShotGather.scale(0.0)
	superShotGatherNp=superShotGather.getNdArray()

	# Copy value to super shot gather
	if (offset=="pos"):
		for iRec in range(nRecNew):
			for its in range(nts):
				superShotGatherNp[0][iRec][its]=obsDataNp[iShot][iShotRecGrid+nRecNew-iRec][its]
				superShotGatherNp[0][iRec+nRecNew+1][its]=modelDataNp[iShot][iShotRecGrid+iRec+1][its]
		superShotGatherNp[0][nRecNew][:]=obsDataNp[iShot][iShotRecGrid][:]

	else:
		for iRec in range(nRecNew):
			superShotGatherNp[0][iRec][:]=obsDataNp[iShot][iRec][:]
			superShotGatherNp[0][iRec+nRecNew+1][:]=modelDataNp[iShot][iShotRecGrid-iRec-1][:]
		superShotGatherNp[0][nRecNew][:]=obsDataNp[iShot][iShotRecGrid][:]

	# Write super shot gather
	superShotGatherFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(superShotGatherFile,superShotGather)

#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# operators
import Elastic_iso_float_we
import SpaceInterpMultiFloat
import StaggerFloat
import PadTruncateSourceFloat
import pyOperator as Op


def forcing_term_op_init(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Interp operator init
	zCoord,xCoord,centerHyper = SpaceInterpMultiFloat.space_interp_multi_init_source(args)

	#interp operator instantiate
	#check which source injection interp method
	sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)
	nt = parObject.getInt("nts")
	spaceInterpMultiOp = SpaceInterpMultiFloat.space_interp_multi(zCoord,xCoord,centerHyper,nt,sourceInterpMethod,sourceInterpNumFilters)


	# pad truncate init
	dt = parObject.getFloat("dts",0.0)
	nExp = parObject.getInt("nExp")
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dt)
	wfldAxis=Hypercube.axis(n=5,o=0.0,d=1)
	regSourceAxis=Hypercube.axis(n=spaceInterpMultiOp.getNDeviceReg(),o=0.0,d=1)
	irregSourceAxis=Hypercube.axis(n=spaceInterpMultiOp.getNDeviceIrreg(),o=0.0,d=1)
	regSourceHyper=Hypercube.hypercube(axes=[regSourceAxis,wfldAxis,tAxis])
	irregSourceHyper=Hypercube.hypercube(axes=[irregSourceAxis,wfldAxis,tAxis])
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),wfldAxis,tAxis])

	input = SepVector.getSepVector(irregSourceHyper,storage="dataFloat")
	padTruncateDummyModel = SepVector.getSepVector(regSourceHyper,storage="dataFloat")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataFloat")
	sourceGridPositions = spaceInterpMultiOp.getRegPosUniqueVector()

	padTruncateSourceOp = PadTruncateSourceFloat.pad_truncate_source(padTruncateDummyModel,padTruncateDummyData,sourceGridPositions)

	#stagger op
	# staggerDummyModel = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataFloat")
	# output = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataFloat")
	# wavefieldStaggerOp=StaggerFloat.stagger_wfld(staggerDummyModel,output)

	#chain operators
	spaceInterpMultiOp.setDomainRange(padTruncateDummyModel,input)
	spaceInterpMultiOp = Op.Transpose(spaceInterpMultiOp)
	PK_adj = Op.ChainOperator(spaceInterpMultiOp,padTruncateSourceOp)
	#SPK_adj = Op.ChainOperator(PK_adj,wavefieldStaggerOp)

	#read in source
	# waveletFloat = SepVector.getSepVector(SPK_adj.getDomain().getHyper(),storage="dataFloat")
	priorData = SepVector.getSepVector(PK_adj.getRange().getHyper(),storage="dataFloat")
	priorModel = SepVector.getSepVector(PK_adj.getDomain().getHyper(),storage="dataFloat")
	waveletFile=parObject.getString("wavelet")
	waveletFloat=genericIO.defaultIO.getVector(waveletFile)
	waveletSMat=waveletFloat.getNdArray()
	waveletSMatT=np.transpose(waveletSMat)
	priorModelMat=priorModel.getNdArray()
	#loop over irreg grid sources and set each to wavelet
	for iShot in range(irregSourceAxis.n):
		priorModelMat[:,:,iShot] = waveletSMatT

	PK_adj.forward(False,priorModel,priorData)

	return PK_adj,priorData

#
def data_extraction_op_init(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Interp operator init
	zCoord,xCoord,centerHyper = SpaceInterpMultiFloat.space_interp_multi_init_rec(args)

	# Horizontal axis
	nx=centerHyper.getAxis(2).n
	dx=centerHyper.getAxis(2).d
	ox=centerHyper.getAxis(2).o

	# Vertical axis
	nz=centerHyper.getAxis(1).n
	dz=centerHyper.getAxis(1).d
	oz=centerHyper.getAxis(1).o

	#interp operator instantiate
	#check which rec injection interp method
	recInterpMethod = parObject.getString("recInterpMethod","linear")
	recInterpNumFilters = parObject.getInt("recInterpNumFilters",4)
	nt = parObject.getInt("nts")
	spaceInterpMultiOp = SpaceInterpMultiFloat.space_interp_multi(zCoord,xCoord,centerHyper,nt,recInterpMethod,recInterpNumFilters)

	# pad truncate init
	dts = parObject.getFloat("dts",0.0)
	nExp = parObject.getInt("nExp")
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dts)
	wfldAxis=Hypercube.axis(n=5,o=0.0,d=1)
	regRecAxis=Hypercube.axis(n=spaceInterpMultiOp.getNDeviceReg(),o=0.0,d=1)
	oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("fat")
	dxReceiver=parObject.getInt("dReceiver")
	irregRecAxis=Hypercube.axis(n=spaceInterpMultiOp.getNDeviceIrreg(),o=ox+oxReceiver*dx,d=dxReceiver*dx)
	regRecHyper=Hypercube.hypercube(axes=[regRecAxis,wfldAxis,tAxis])
	irregRecHyper=Hypercube.hypercube(axes=[irregRecAxis,wfldAxis,tAxis])
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),wfldAxis,tAxis])

	output = SepVector.getSepVector(irregRecHyper,storage="dataFloat")
	padTruncateDummyModel = SepVector.getSepVector(regRecHyper,storage="dataFloat")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataFloat")
	recGridPositions = spaceInterpMultiOp.getRegPosUniqueVector()
	padTruncateRecOp = PadTruncateSourceFloat.pad_truncate_source(padTruncateDummyModel,padTruncateDummyData,recGridPositions)
	padTruncateRecOp = Op.Transpose(padTruncateRecOp)

	#stagger op
	staggerDummyModel = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataFloat")
	input = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataFloat")
	wavefieldStaggerOp=StaggerFloat.stagger_wfld(staggerDummyModel,input)
	wavefieldStaggerOp=Op.Transpose(wavefieldStaggerOp)

	#chain operators
	spaceInterpMultiOp.setDomainRange(padTruncateDummyModel,output)
	#spaceInterpMultiOp = Op.Transpose(spaceInterpMultiOp)
	# P_adjS_adj = Op.ChainOperator(wavefieldStaggerOp,padTruncateRecOp)
	KP_adj = Op.ChainOperator(P_adj,spaceInterpMultiOp)

	#apply forward
	return KP_adj

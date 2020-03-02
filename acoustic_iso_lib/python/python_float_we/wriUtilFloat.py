#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os
import math

# operators
import Acoustic_iso_float_we
import TruncateSpatialReg
import SpaceInterpFloat
import PadTruncateSource
import SampleWfld
# import GF
import SphericalSpreadingScale
import pyOperator as pyOp
import scipy as sp
import scipy.ndimage
import fft_wfld

def fft_wfld_init(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Time and freq Axes
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)

	freq_np_axis = np.fft.rfftfreq(nts,dts)
	df = freq_np_axis[1]
	nf= freq_np_axis.size
	# nf=nts//2+1 # number of samples in Fourier domain
	# fs=1/dts #sampling rate of time domain
	# if(nts%2 == 0): #even input
	# 	f_range = fs/2
	# else: # odd input
	# 	f_range = fs/2*(nts-1)/nts
	# df = f_range/nf
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	wAxis=Hypercube.axis(n=nf,o=0,d=df)

	# z Axis model
	nz=parObject.getInt("nz",-1)
	oz=parObject.getFloat("oz",-1.0)
	dz=parObject.getFloat("dz",-1.0)
	zAxis=Hypercube.axis(n=nz,o=oz,d=dz)

	# x axis model
	nx=parObject.getInt("nx",-1)
	ox=parObject.getFloat("ox",-1.0)
	dx=parObject.getFloat("dx",-1.0)
	xAxis=Hypercube.axis(n=nx,o=ox,d=dx)

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# Allocate model
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis,wAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataComplex")

	# init op
	op = fft_wfld.fft_wfld(modelFloat,dataFloat)

	#apply forward
	return modelFloat,dataFloat,op

def fft_wfld_multi_exp_init(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Time and freq Axes
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)

	freq_np_axis = np.fft.rfftfreq(nts,dts)
	df = freq_np_axis[1]
	nf= freq_np_axis.size
	# nf=nts//2+1 # number of samples in Fourier domain
	# fs=1/dts #sampling rate of time domain
	# if(nts%2 == 0): #even input
	# 	f_range = fs/2
	# else: # odd input
	# 	f_range = fs/2*(nts-1)/nts
	# df = f_range/nf
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	wAxis=Hypercube.axis(n=nf,o=0,d=df)

	fmax=parObject.getFloat("fmax",-1.0)
	if(fmax!=-1.0):
		nf_wind=min(int(fmax/df)+1,nf)
		print("\tfmax selected: ",fmax)
		print("\nnew nf selected: ",nf_wind)
		wAxis_wind=Hypercube.axis(n=nf_wind,o=0,d=df)

	# z Axis model
	nz=parObject.getInt("nz",-1)
	oz=parObject.getFloat("oz",-1.0)
	dz=parObject.getFloat("dz",-1.0)
	zAxis=Hypercube.axis(n=nz,o=oz,d=dz)

	# x axis model
	nx=parObject.getInt("nx",-1)
	ox=parObject.getFloat("ox",-1.0)
	dx=parObject.getFloat("dx",-1.0)
	xAxis=Hypercube.axis(n=nx,o=ox,d=dx)

	# shot Axis
	nExp=parObject.getInt("nExp",1)
	shotAxis=Hypercube.axis(n=nExp,o=0,d=1)

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis,shotAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# Allocate model
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis,wAxis,shotAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataComplex")

	# init op
	fftOpTemp = fft_wfld.fft_wfld(modelFloat,dataFloat,axis=1)
	if(fmax!=-1.0):
		modelHyper_wind=Hypercube.hypercube(axes=[zAxis,xAxis,wAxis_wind,shotAxis])
		modelFloat_wind=SepVector.getSepVector(modelHyper_wind,storage="dataComplex")
		padTruncateFreqOp=PadTruncateSource.zero_pad_4d(modelFloat_wind,modelFloat)
		fftOp=fftOpTemp*padTruncateFreqOp
		modelFloat=modelFloat_wind
	else:
		fftOp=fftOpTemp

	#apply forward
	return modelFloat,dataFloat,fftOp

def grad_edit_mora(gradNdArray):
	sigma_y=1
	sigma_x=1
	sigma = [sigma_y, sigma_x]

	gradNdArray[:,0:31]=0
	gradNdArrayOut = sp.ndimage.filters.gaussian_filter(gradNdArray, sigma, mode='constant')
	return gradNdArrayOut

def grad_edit_diving(gradNdArray):
	sigma_y=14
	sigma_x=12
	sigma = [sigma_y, sigma_x]

	#gradNdArray[:,0:42]=0
	gradNdArrayOut = sp.ndimage.filters.gaussian_filter(gradNdArray, sigma, mode='constant')
	gradNdArrayOut[:,0:40]=0
	return gradNdArrayOut

def evaluate_epsilon(current_p_model,p_dataFloat,prior,dataSamplingOp,waveEquationAcousticOp,parObject):
	#make first data residual
	K_resid = dataSamplingOp.getRange().clone()
	dataSamplingOp.forward(0,current_p_model,K_resid)
	K_resid.scaleAdd(p_dataFloat,1,-1) # Kp-d

	#make first model residual
	A_resid = waveEquationAcousticOp.getRange().clone()
	waveEquationAcousticOp.forward(1,current_p_model,A_resid)
	A_resid.scaleAdd(prior,1,-1) # Ap-f

	if(current_p_model.norm()==0): #if initial model is zero, take a step before balancing eps
		#update model
		modelOne = current_p_model.clone()
		modelOne.zero()

		dataSamplingOp.adjoint(1,modelOne,K_resid)

		waveEquationAcousticOp.adjoint(1,modelOne,A_resid)

		dataSamplingOp.forward(0,modelOne,K_resid)
		K_resid.scaleAdd(p_dataFloat,1,-1)
		waveEquationAcousticOp.forward(0,modelOne,A_resid)
		A_resid.scaleAdd(prior,1,-1) # Ap-f


	epsilon_p = math.sqrt(K_resid.dot(K_resid)/A_resid.dot(A_resid))

	return epsilon_p

def forcing_term_op_init_p(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Interp operator init
	zCoord,xCoord,centerHyper = SpaceInterpFloat.space_interp_init_source(args)

	#interp operator instantiate
	#check which source injection interp method
	sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)
	nt = parObject.getInt("nts")
	spaceInterpOp = SpaceInterpFloat.space_interp(zCoord,xCoord,centerHyper,nt,sourceInterpMethod,sourceInterpNumFilters)


	# pad truncate init
	dt = parObject.getFloat("dts",0.0)
	nExp = parObject.getInt("nExp")
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dt)
	regSourceAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceReg(),o=0.0,d=1)
	irregSourceAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceIrreg(),o=0.0,d=1)
	regSourceHyper=Hypercube.hypercube(axes=[regSourceAxis,tAxis])
	irregSourceHyper=Hypercube.hypercube(axes=[irregSourceAxis,tAxis])
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),tAxis])

	input = SepVector.getSepVector(irregSourceHyper,storage="dataFloat")
	padTruncateDummyModel = SepVector.getSepVector(regSourceHyper,storage="dataFloat")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataFloat")
	sourceGridPositions = spaceInterpOp.getRegPosUniqueVector()

	padTruncateSourceOp = PadTruncateSource.pad_truncate_source(padTruncateDummyModel,padTruncateDummyData,sourceGridPositions)

	#stagger op
	# staggerDummyModel = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataFloat")
	# output = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataFloat")
	# wavefieldStaggerOp=StaggerFloat.stagger_wfld(staggerDummyModel,output)

	#chain operators
	spaceInterpOp.setDomainRange(padTruncateDummyModel,input)
	spaceInterpOp = pyOp.Transpose(spaceInterpOp)
	PK_adj = pyOp.ChainOperator(spaceInterpOp,padTruncateSourceOp)
	#SPK_adj = pyOp.ChainOperator(PK_adj,wavefieldStaggerOp)

	#read in source
	# waveletFloat = SepVector.getSepVector(SPK_adj.getDomain().getHyper(),storage="dataFloat")
	priorData = SepVector.getSepVector(PK_adj.getRange().getHyper(),storage="dataFloat")
	priorModel = SepVector.getSepVector(PK_adj.getDomain().getHyper(),storage="dataFloat")
	waveletFile=parObject.getString("wavelet","None")
	if(waveletFile=="None"): waveletFile=parObject.getString("wavelet_p","None")
	waveletFloat=genericIO.defaultIO.getVector(waveletFile)
	waveletSMat=waveletFloat.getNdArray()
	waveletSMatT=np.transpose(waveletSMat)
	priorModelMat=priorModel.getNdArray()
	#loop over irreg grid sources and set each to wavelet
	for iShot in range(irregSourceAxis.n):
		priorModelMat[:,iShot] = waveletSMatT


	PK_adj.forward(False,priorModel,priorData)
	#spaceInterpOp.forward(0,priorModel,padTruncateDummyModel)

	return PK_adj,priorData

#forcing term init over mutliple shots
def forcing_term_op_init_p_multi_exp(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Interp operator init
	xCoord,zCoord,experimentId,centerHyper = SpaceInterpFloat.space_interp_init_source_multi_exp(args)
	#interp operator instantiate
	#check which source injection interp method
	sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)
	nt = parObject.getInt("nts")
	spaceInterpOp = SpaceInterpFloat.space_interp_multi_exp(zCoord,xCoord,experimentId,centerHyper,nt,sourceInterpMethod,sourceInterpNumFilters)

	# pad truncate init
	dt = parObject.getFloat("dts",0.0)
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dt)
	nExp = len(spaceInterpOp.getRegPosUniqueVector())
	regSourceAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceReg(),o=0.0,d=1)
	irregSourceAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceIrreg(),o=0.0,d=1)
	regSourceHyper=Hypercube.hypercube(axes=[regSourceAxis,tAxis])
	irregSourceHyper=Hypercube.hypercube(axes=[irregSourceAxis,tAxis])
	expAxis=Hypercube.axis(n=nExp,o=0.0,d=1)
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),tAxis,expAxis])

	input = SepVector.getSepVector(irregSourceHyper,storage="dataFloat")
	padTruncateDummyModel = SepVector.getSepVector(regSourceHyper,storage="dataFloat")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataFloat")
	sourceGridPositions = spaceInterpOp.getRegPosUniqueVector()
	sourceGridExpMapping = spaceInterpOp.getIndexMaps()

	padTruncateSourceOp = PadTruncateSource.pad_truncate_source_multi_exp(padTruncateDummyModel,padTruncateDummyData,sourceGridPositions,sourceGridExpMapping)

	#chain operators
	spaceInterpOp.setDomainRange(padTruncateDummyModel,input)
	spaceInterpOp = spaceInterpOp.T
	PK_adj = pyOp.ChainOperator(spaceInterpOp,padTruncateSourceOp)


	#read in source
	priorData = SepVector.getSepVector(PK_adj.getRange().getHyper(),storage="dataFloat")
	priorModel = SepVector.getSepVector(PK_adj.getDomain().getHyper(),storage="dataFloat")


	waveletFile=parObject.getString("wavelet","None")
	if(waveletFile=="None"): waveletFile=parObject.getString("wavelet_p","None")
	waveletFloat=genericIO.defaultIO.getVector(waveletFile)
	waveletSMat=waveletFloat.getNdArray()
	waveletSMatT=np.transpose(waveletSMat)
	priorModelMat=priorModel.getNdArray()
	#loop over irreg grid sources and set each to wavelet
	for iShot in range(irregSourceAxis.n):
		priorModelMat[:,iShot] = waveletSMatT

	# print('here1')
	# print(priorModel.getNdArray().shape)
	# print(priorData.getNdArray().shape)
	# print("spaceInterpOp domain: ", spaceInterpOp.getDomain().getNdArray().shape)
	# print("spaceInterpOp range: ", spaceInterpOp.getRange().getNdArray().shape)
	# print("padTruncateSourceOp domain: ", padTruncateSourceOp.getDomain().getNdArray().shape)
	# print("padTruncateSourceOp range: ", padTruncateSourceOp.getRange().getNdArray().shape)
	PK_adj.forward(False,priorModel,priorData)
	#spaceInterpOp.forward(0,priorModel,padTruncateDummyModel)

	return PK_adj,priorData

def forcing_term_op_init_m(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Interp operator init
	zCoord,xCoord,centerHyper = SpaceInterpFloat.space_interp_init_source(args)

	#interp operator instantiate
	#check which source injection interp method
	sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)
	nt = parObject.getInt("nts")
	spaceInterpOp = SpaceInterpFloat.space_interp(zCoord,xCoord,centerHyper,nt,sourceInterpMethod,sourceInterpNumFilters)


	# pad truncate init
	dt = parObject.getFloat("dts",0.0)
	nExp = parObject.getInt("nExp")
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dt)
	regSourceAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceReg(),o=0.0,d=1)
	irregSourceAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceIrreg(),o=0.0,d=1)
	regSourceHyper=Hypercube.hypercube(axes=[regSourceAxis,tAxis])
	irregSourceHyper=Hypercube.hypercube(axes=[irregSourceAxis,tAxis])
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),tAxis])

	input = SepVector.getSepVector(irregSourceHyper,storage="dataFloat")
	padTruncateDummyModel = SepVector.getSepVector(regSourceHyper,storage="dataFloat")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataFloat")
	sourceGridPositions = spaceInterpOp.getRegPosUniqueVector()

	padTruncateSourceOp = PadTruncateSource.pad_truncate_source(padTruncateDummyModel,padTruncateDummyData,sourceGridPositions)

	#stagger op
	# staggerDummyModel = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataFloat")
	# output = SepVector.getSepVector(padTruncateDummyData.getHyper(),storage="dataFloat")
	# wavefieldStaggerOp=StaggerFloat.stagger_wfld(staggerDummyModel,output)

	#chain operators
	spaceInterpOp.setDomainRange(padTruncateDummyModel,input)
	spaceInterpOp = spaceInterpOp.T
	PK_adj = pyOp.ChainOperator(spaceInterpOp,padTruncateSourceOp)
	#SPK_adj = pyOp.ChainOperator(PK_adj,wavefieldStaggerOp)

	#read in source
	# waveletFloat = SepVector.getSepVector(SPK_adj.getDomain().getHyper(),storage="dataFloat")
	priorData = SepVector.getSepVector(PK_adj.getRange().getHyper(),storage="dataFloat")
	priorModel = SepVector.getSepVector(PK_adj.getDomain().getHyper(),storage="dataFloat")
	waveletFile=parObject.getString("wavelet","None")
	if(waveletFile=="None"): waveletFile=parObject.getString("wavelet_m","None")
	waveletFloat=genericIO.defaultIO.getVector(waveletFile)
	waveletSMat=waveletFloat.getNdArray()
	waveletSMatT=np.transpose(waveletSMat)
	priorModelMat=priorModel.getNdArray()
	#loop over irreg grid sources and set each to wavelet
	for iShot in range(irregSourceAxis.n):
		priorModelMat[:,iShot] = waveletSMatT


	PK_adj.forward(False,priorModel,priorData)
	#spaceInterpOp.forward(0,priorModel,padTruncateDummyModel)

	return PK_adj,priorData

#forcing term init over mutliple shots
def forcing_term_op_init_m_mutli_exp(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Interp operator init
	xCoord,zCoord,experimentId,centerHyper = SpaceInterpFloat.space_interp_init_source_multi_exp(args)
	#interp operator instantiate
	#check which source injection interp method
	sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)
	nt = parObject.getInt("nts")
	spaceInterpOp = SpaceInterpFloat.space_interp_multi_exp(zCoord,xCoord,experimentId,centerHyper,nt,sourceInterpMethod,sourceInterpNumFilters)

	# pad truncate init
	dt = parObject.getFloat("dts",0.0)
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dt)
	nExp = len(spaceInterpOp.getRegPosUniqueVector())
	regSourceAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceReg(),o=0.0,d=1)
	irregSourceAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceIrreg(),o=0.0,d=1)
	regSourceHyper=Hypercube.hypercube(axes=[regSourceAxis,tAxis])
	irregSourceHyper=Hypercube.hypercube(axes=[irregSourceAxis,tAxis])
	expAxis=Hypercube.axis(n=nExp,o=0.0,d=1)
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),tAxis,expAxis])

	input = SepVector.getSepVector(irregSourceHyper,storage="dataFloat")
	padTruncateDummyModel = SepVector.getSepVector(regSourceHyper,storage="dataFloat")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataFloat")
	sourceGridPositions = spaceInterpOp.getRegPosUniqueVector()
	sourceGridExpMapping = spaceInterpOp.getIndexMaps()

	padTruncateSourceOp = PadTruncateSource.pad_truncate_source_multi_exp(padTruncateDummyModel,padTruncateDummyData,sourceGridPositions,sourceGridExpMapping)

	#chain operators
	spaceInterpOp.setDomainRange(padTruncateDummyModel,input)
	spaceInterpOp = spaceInterpOp.T
	PK_adj = pyOp.ChainOperator(spaceInterpOp,padTruncateSourceOp)
	#SPK_adj = pyOp.ChainOperator(PK_adj,wavefieldStaggerOp)

	#read in source
	# waveletFloat = SepVector.getSepVector(SPK_adj.getDomain().getHyper(),storage="dataFloat")
	priorData = SepVector.getSepVector(PK_adj.getRange().getHyper(),storage="dataFloat")
	priorModel = SepVector.getSepVector(PK_adj.getDomain().getHyper(),storage="dataFloat")


	waveletFile=parObject.getString("wavelet","None")
	if(waveletFile=="None"): waveletFile=parObject.getString("wavelet_m","None")
	waveletFloat=genericIO.defaultIO.getVector(waveletFile)
	waveletSMat=waveletFloat.getNdArray()
	waveletSMatT=np.transpose(waveletSMat)
	priorModelMat=priorModel.getNdArray()
	#loop over irreg grid sources and set each to wavelet
	for iShot in range(irregSourceAxis.n):
		priorModelMat[:,iShot] = waveletSMatT

	PK_adj.forward(False,priorModel,priorData)
	#spaceInterpOp.forward(0,priorModel,padTruncateDummyModel)

	return PK_adj,priorData

def spherical_spreading_op_init(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	#get source locations
	zCoordSou,xCoordSou,centerHyper = SpaceInterpFloat.space_interp_init_source(args)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# z Axis model
	nz=parObject.getInt("nz",-1)
	oz=parObject.getFloat("oz",-1.0)
	dz=parObject.getFloat("dz",-1.0)
	zAxis=Hypercube.axis(n=nz,o=oz,d=dz)

	# x axis model
	nx=parObject.getInt("nx",-1)
	ox=parObject.getFloat("ox",-1.0)
	dx=parObject.getFloat("dx",-1.0)
	xAxis=Hypercube.axis(n=nx,o=ox,d=dx)

	# Allocate model
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# get min vel
	minvel = parObject.getFloat("minVel", 1500)

	# get tpow precond value
	tpow=parObject.getFloat("tpowPrecond",0.0)

	# init op
	op = SphericalSpreadingScale.spherical_spreading_scale(modelFloat,dataFloat,zCoordSou,xCoordSou,tpow,minvel)

	#apply forward
	return modelFloat,dataFloat,op
#
# def greens_function_op_init(args):
#
# 	# Bullshit stuff
# 	parObject=genericIO.io(params=sys.argv)
#
# 	#get source locations
# 	zCoordSou,xCoordSou,centerHyper = SpaceInterpFloat.space_interp_init_source(args)
#
# 	# Time Axis
# 	nts=parObject.getInt("nts",-1)
# 	ots=parObject.getFloat("ots",0.0)
# 	dts=parObject.getFloat("dts",-1.0)
# 	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
#
# 	# z Axis model
# 	nz=parObject.getInt("nz",-1)
# 	oz=parObject.getFloat("oz",-1.0)
# 	dz=parObject.getFloat("dz",-1.0)
# 	zAxis=Hypercube.axis(n=nz,o=oz,d=dz)
#
# 	# x axis model
# 	nx=parObject.getInt("nx",-1)
# 	ox=parObject.getFloat("ox",-1.0)
# 	dx=parObject.getFloat("dx",-1.0)
# 	xAxis=Hypercube.axis(n=nx,o=ox,d=dx)
#
# 	# Allocate model
# 	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
# 	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")
#
# 	# Allocate data
# 	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
# 	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")
#
# 	# calculate max velocity value
# 	slsqFile=parObject.getString("slsq", "noVpFile")
# 	if (slsqFile == "noVpFile"):
# 		print("**** ERROR: User did not provide slsq file, slsq ****\n")
# 		sys.exit()
# 	slsq=genericIO.defaultIO.getVector(slsqFile)
# 	slsqNdArray = slsq.getNdArray()
# 	slsqNonzero = slsqNdArray[slsqNdArray>0]
# 	minslsq = slsqNonzero.min()
# 	maxvel = math.sqrt(1/minslsq)
# 	print("maxvel: ", maxvel)
#
#
# 	tstart=parObject.getFloat("t_start", 0.0)
#
# 	# init op
# 	op = GF.gf(modelFloat,dataFloat,zCoordSou,xCoordSou,tstart,maxvel)
#
# 	#apply forward
# 	return modelFloat,dataFloat,op
def wfld_extraction_reg_op_init(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	#get source locations
	zCoordSou,xCoordSou,centerHyper = SpaceInterpFloat.space_interp_init_source(args)
	#get rec locations
	zCoordRec,xCoordRec,_ = SpaceInterpFloat.space_interp_init_rec(args)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# z Axis model
	nz=parObject.getInt("nz",-1)
	oz=parObject.getFloat("oz",-1.0)
	dz=parObject.getFloat("dz",-1.0)
	zAxis=Hypercube.axis(n=nz,o=oz,d=dz)

	# x axis model
	nx=parObject.getInt("nx",-1)
	ox=parObject.getFloat("ox",-1.0)
	dx=parObject.getFloat("dx",-1.0)
	xAxis=Hypercube.axis(n=nx,o=ox,d=dx)

	# Allocate model
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# calculate max velocity value
	slsqFile=parObject.getString("slsq", "noVpFile")
	if (slsqFile == "noVpFile"):
		print("**** ERROR: User did not provide slsq file, slsq ****\n")
		sys.exit()
	slsq=genericIO.defaultIO.getVector(slsqFile)
	minslsq = slsq.getNdArray().min()
	maxvel = math.sqrt(1/minslsq)
	print("maxvel: ", maxvel)


	tstart=parObject.getFloat("t_start", 0.0)

	# init op
	op = SampleWfld.sample_wfld(modelFloat,dataFloat,zCoordSou,xCoordSou,zCoordRec,xCoordRec,tstart,maxvel)

	#apply forward
	return modelFloat,dataFloat,op
#
def data_extraction_reg_op_init(args):

	# IO stuff
	parObject=genericIO.io(params=sys.argv)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# z Axis model
	nz=parObject.getInt("nz",-1)
	oz=parObject.getFloat("oz",-1.0)
	dz=parObject.getFloat("dz",-1.0)
	zAxis=Hypercube.axis(n=nz,o=oz,d=dz)

	# x axis model
	nx=parObject.getInt("nx",-1)
	ox=parObject.getFloat("ox",-1.0)
	dx=parObject.getFloat("dx",-1.0)
	xAxis=Hypercube.axis(n=nx,o=ox,d=dx)

	# Allocate model
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# z Axis data
	nzData=parObject.getInt("nzData",-1)
	ozData=parObject.getFloat("ozData",-1.0)
	dzData=parObject.getFloat("dzData",-1.0)
	zAxisData=Hypercube.axis(n=nzData,o=ozData,d=dzData)

	# x axis data
	nxData=parObject.getInt("nxData",-1)
	oxData=parObject.getFloat("oxData",-1.0)
	dxData=parObject.getFloat("dxData",-1.0)
	xAxisData=Hypercube.axis(n=nxData,o=oxData,d=dxData)

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxisData,xAxisData,timeAxis])
	dataFloat = SepVector.getSepVector(dataHyper,storage="dataFloat")

	# init op
	op=TruncateSpatialReg.sampleDataReg(modelFloat,dataFloat)

	#apply forward
	return modelFloat,dataFloat,op

def data_extraction_op_init_multi_exp_freq(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Interp operator init
	xCoord,zCoord,experimentId,centerHyper = SpaceInterpFloat.space_interp_init_rec_multi_exp(args)

	# Horizontal axis
	nx=centerHyper.getAxis(2).n
	dx=centerHyper.getAxis(2).d
	ox=centerHyper.getAxis(2).o

	# Vertical axis
	nz=centerHyper.getAxis(1).n
	dz=centerHyper.getAxis(1).d
	oz=centerHyper.getAxis(1).o

	# Time and freq Axes
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)

	freq_np_axis = np.fft.rfftfreq(nts,dts)
	df = freq_np_axis[1]
	nf= freq_np_axis.size
	# nf=nts//2+1 # number of samples in Fourier domain
	# fs=1/dts #sampling rate of time domain
	# if(nts%2 == 0): #even input
	# 	f_range = fs/2
	# else: # odd input
	# 	f_range = fs/2*(nts-1)/nts
	# df = f_range/nf
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	wAxis_orig=Hypercube.axis(n=nf,o=0,d=df)

	fmax=parObject.getFloat("fmax",-1.0)
	if(fmax!=-1.0):
		nf_wind=min(int(fmax/df)+1,nf)
		print("\tfmax selected: ",fmax)
		print("\tnew nf selected: ",nf_wind)
		wAxis_wind=Hypercube.axis(n=nf_wind,o=0,d=df)
		wAxis=wAxis_wind
		nf=nf_wind
	else:
		wAxis=wAxis_orig
	# print('nts:'+str(nts)+' ots:'+str(ots)+' dts:' + str(dts))
	# print('nf:'+str(nf)+'f_range: '+ str(f_range) + ' of:0 df:' + str(df))

	#interp operator instantiate
	#check which rec injection interp method
	recInterpMethod = parObject.getString("recInterpMethod","linear")
	recInterpNumFilters = parObject.getInt("recInterpNumFilters",4)

	spaceInterpOp = SpaceInterpFloat.space_interp_multi_exp_complex(zCoord,xCoord,experimentId,centerHyper,nf,recInterpMethod,recInterpNumFilters)
	# pad truncate init
	tAxis=Hypercube.axis(n=nts,o=0.0,d=dts)
	nExp = len(spaceInterpOp.getRegPosUniqueVector())

	regReceiverAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceReg(),o=0.0,d=1)
	irregReceiverAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceIrreg(),o=0.0,d=1)
	expAxis=Hypercube.axis(n=nExp,o=0.0,d=1)

	regReceiverHyper=Hypercube.hypercube(axes=[regReceiverAxis,wAxis])
	irregReceiverHyper=Hypercube.hypercube(axes=[irregReceiverAxis,wAxis])
	irregReceiverHyperTime=Hypercube.hypercube(axes=[irregReceiverAxis,tAxis])
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),wAxis,expAxis])

	output = SepVector.getSepVector(irregReceiverHyper,storage="dataComplex")
	outputTime= SepVector.getSepVector(irregReceiverHyperTime,storage="dataFloat")
	padTruncateDummyModel = SepVector.getSepVector(regReceiverHyper,storage="dataComplex")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataComplex")
	sourceGridPositions = spaceInterpOp.getRegPosUniqueVector()
	sourceGridExpMapping = spaceInterpOp.getIndexMaps()

	padTruncateReceiverOp = PadTruncateSource.pad_truncate_source_multi_exp_complex(padTruncateDummyModel,padTruncateDummyData,sourceGridPositions,sourceGridExpMapping)
	padTruncateRecOp = padTruncateReceiverOp.T

	#chain operators
	spaceInterpOp.setDomainRange(padTruncateDummyModel,output)

	# init fft op
	#fftOpTemp = fft_wfld.fft_wfld(output,outputTime,axis=0)
	# init fft op
	#fftOpTemp = fft_wfld.fft_wfld(modelFloat,timeFloat,axis=1)
	if(fmax!=-1.0):
		irregReceiverHyper_orig=Hypercube.hypercube(axes=[irregReceiverAxis,wAxis_orig])
		output_orig = SepVector.getSepVector(irregReceiverHyper_orig,storage="dataComplex")
		padTruncateFreqOp=PadTruncateSource.zero_pad_2d(output,output_orig)
		fftOpTemp = fft_wfld.fft_wfld(output_orig,outputTime,axis=0)
		fftOp=pyOp.ChainOperator(padTruncateFreqOp,fftOpTemp)
	else:
		fftOp = fft_wfld.fft_wfld(output,outputTime,axis=0)

	KP_adj = pyOp.ChainOperator(padTruncateRecOp,spaceInterpOp)

	#apply forward
	return padTruncateDummyData,output,KP_adj,fftOp,outputTime

def data_extraction_op_init_multi_exp(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Interp operator init
	xCoord,zCoord,experimentId,centerHyper = SpaceInterpFloat.space_interp_init_rec_multi_exp(args)

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

	spaceInterpOp = SpaceInterpFloat.space_interp_multi_exp(zCoord,xCoord,experimentId,centerHyper,nt,recInterpMethod,recInterpNumFilters)
	# pad truncate init
	dt = parObject.getFloat("dts",0.0)
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dt)
	nExp = len(spaceInterpOp.getRegPosUniqueVector())
	regReceiverAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceReg(),o=0.0,d=1)
	irregReceiverAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceIrreg(),o=0.0,d=1)
	regReceiverHyper=Hypercube.hypercube(axes=[regReceiverAxis,tAxis])
	irregReceiverHyper=Hypercube.hypercube(axes=[irregReceiverAxis,tAxis])
	expAxis=Hypercube.axis(n=nExp,o=0.0,d=1)
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),tAxis,expAxis])

	output = SepVector.getSepVector(irregReceiverHyper,storage="dataFloat")
	padTruncateDummyModel = SepVector.getSepVector(regReceiverHyper,storage="dataFloat")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataFloat")
	sourceGridPositions = spaceInterpOp.getRegPosUniqueVector()
	sourceGridExpMapping = spaceInterpOp.getIndexMaps()

	padTruncateReceiverOp = PadTruncateSource.pad_truncate_source_multi_exp(padTruncateDummyModel,padTruncateDummyData,sourceGridPositions,sourceGridExpMapping)
	padTruncateRecOp = padTruncateReceiverOp.T

	#chain operators
	spaceInterpOp.setDomainRange(padTruncateDummyModel,output)

	KP_adj = pyOp.ChainOperator(padTruncateRecOp,spaceInterpOp)

	#apply forward
	return padTruncateDummyData,output,KP_adj

def data_extraction_op_init(args):

	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Interp operator init
	zCoord,xCoord,centerHyper = SpaceInterpFloat.space_interp_init_rec(args)

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
	spaceInterpOp = SpaceInterpFloat.space_interp(zCoord,xCoord,centerHyper,nt,recInterpMethod,recInterpNumFilters)

	# pad truncate init
	dts = parObject.getFloat("dts",0.0)
	nExp = parObject.getInt("nExp")
	tAxis=Hypercube.axis(n=nt,o=0.0,d=dts)
	regRecAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceReg(),o=0.0,d=1)
	oxReceiver=parObject.getInt("oxReceiver")-1+parObject.getInt("xPadMinus",0)+parObject.getInt("fat")
	dxReceiver=parObject.getInt("dxReceiver")
	irregRecAxis=Hypercube.axis(n=spaceInterpOp.getNDeviceIrreg(),o=(oxReceiver)*dx+ox,d=dxReceiver*dx)
	regRecHyper=Hypercube.hypercube(axes=[regRecAxis,tAxis])
	irregRecHyper=Hypercube.hypercube(axes=[irregRecAxis,tAxis])
	regWfldHyper=Hypercube.hypercube(axes=[centerHyper.getAxis(1),centerHyper.getAxis(2),tAxis])

	output = SepVector.getSepVector(irregRecHyper,storage="dataFloat")
	padTruncateDummyModel = SepVector.getSepVector(regRecHyper,storage="dataFloat")
	padTruncateDummyData = SepVector.getSepVector(regWfldHyper,storage="dataFloat")
	recGridPositions = spaceInterpOp.getRegPosUniqueVector()
	padTruncateRecOp = PadTruncateSource.pad_truncate_source(padTruncateDummyModel,padTruncateDummyData,recGridPositions)
	padTruncateRecOp = padTruncateRecOp.T

	#chain operators
	spaceInterpOp.setDomainRange(padTruncateDummyModel,output)
	KP_adj = pyOp.ChainOperator(padTruncateRecOp,spaceInterpOp)

	#apply forward
	return padTruncateDummyData,output,KP_adj

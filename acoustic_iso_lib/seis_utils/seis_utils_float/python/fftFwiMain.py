#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
# import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Velocity padding
	fat=parObject.getInt("fat",5)
	zPadMinus=parObject.getInt("zPadMinus",100)
	zPadPlus=parObject.getInt("zPadPlus")
	xPadMinus=parObject.getInt("xPadMinus",100)
	xPadPlus=parObject.getInt("xPadPlus")

	# Read true model
	modelTrueFile=parObject.getString("modelTrue")
	modelTrue=genericIO.defaultIO.getVector(modelTrueFile,ndims=2)
	modelTrueNd=modelTrue.getNdArray()

	# Read initial model
	modelInitFile=parObject.getString("modelInit","NoModelInit")
	modeInit=genericIO.defaultIO.getVector(modelInitFile,ndims=2)
	modelInitNd=modeInit.getNdArray()

	# Read inverted models
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=3)

	# Model dimensions
	nz=model.getHyper().axes[0].n
	dz=model.getHyper().axes[0].d
	nx=model.getHyper().axes[1].n
	dx=model.getHyper().axes[1].d
	nIter=model.getHyper().axes[2].n

	# Model dimension without padding
	nzNp=nz-2*fat-zPadMinus-zPadPlus
	nxNp=nx-2*fat-xPadMinus-xPadPlus

	# Data axes
	kz=2*np.pi*np.fft.fftfreq(nzNp,dz)
	kz=np.fft.fftshift(kz)
	dkz=kz[1]-kz[0]
	kx=2*np.pi*np.fft.fftfreq(nxNp,dx)
	kx=np.fft.fftshift(kx)
	dkx=kx[1]-kx[0]

	# Data allocation
	dataZAxis=Hypercube.axis(n=nzNp,o=kz[0],d=dkz,label="Kz [1/km]")
	dataXAxis=Hypercube.axis(n=nxNp,o=kx[0],d=dkx,label="Kx [1/km]")
	dataIterAxis=Hypercube.axis(n=nIter+1,o=0.0,d=1.0,label="Iteration #")
	dataHyper=Hypercube.hypercube(axes=[dataZAxis,dataXAxis,dataIterAxis])
	data=SepVector.getSepVector(dataHyper)

	# Numpy arrays
	modelNd=model.getNdArray()
	dataNd=data.getNdArray()

	# FFT of model updates
	if (modelInitFile=="NoModelInit"):
		temp = np.zeros((nxNp,nzNp))
		for iIter in range(nIter):
			for ix in range(fat+xPadMinus,nx-fat-xPadPlus):
				for iz in range(fat+zPadMinus,nz-fat-zPadPlus):
					temp[ix-fat-xPadMinus][iz-fat-zPadMinus]=modelNd[iIter][ix][iz]-modelNd[0][ix][iz]
			temp=np.fft.fft2(temp)
			dataNd[iIter][:]=np.abs(np.fft.fftshift(temp))

		# FFT of true update
		for ix in range(fat+xPadMinus,nx-fat-xPadPlus):
			for iz in range(fat+zPadMinus,nz-fat-zPadPlus):
				temp[ix-fat-xPadMinus][iz-fat-zPadMinus]=modelTrueNd[ix][iz]-modelNd[0][ix][iz]
		temp=np.fft.fft2(temp)
		dataNd[nIter][:]=np.abs(np.fft.fftshift(temp))

	else:
		temp = np.zeros((nxNp,nzNp))
		for iIter in range(nIter):
			for ix in range(fat+xPadMinus,nx-fat-xPadPlus):
				for iz in range(fat+zPadMinus,nz-fat-zPadPlus):
					temp[ix-fat-xPadMinus][iz-fat-zPadMinus]=modelNd[iIter][ix][iz]-modelInitNd[ix][iz]
			temp=np.fft.fft2(temp)
			dataNd[iIter][:]=np.abs(np.fft.fftshift(temp))

		# FFT of true update
		for ix in range(fat+xPadMinus,nx-fat-xPadPlus):
			for iz in range(fat+zPadMinus,nz-fat-zPadPlus):
				temp[ix-fat-xPadMinus][iz-fat-zPadMinus]=modelTrueNd[ix][iz]-modelInitNd[ix][iz]
		temp=np.fft.fft2(temp)
		dataNd[nIter][:]=np.abs(np.fft.fftshift(temp))

	# Write data amplitude spectrum
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)

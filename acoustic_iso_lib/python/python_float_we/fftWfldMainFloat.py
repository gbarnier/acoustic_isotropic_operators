#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import fft_wfld

# Solver library
import pyOperator as pyOp
import wriUtilFloat
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)

	# get params
	modelFile=parObject.getString("model","noModelFile")
	dataFile=parObject.getString("data","noDataFile")

	# Forward
	if (parObject.getInt("adj",0) == 0):
		print("-------------------------------------------------------------------")
		print("--------- Running Python fft transform of wfld. Transform from freq to time  -----------")
		print("-------------------------------------------------------------------\n")

		# Read  model
		modelFloat=genericIO.defaultIO.getVector(modelFile,storage='dataComplex')

		# Time and freq Axes
		nw=modelFloat.getHyper().getAxis(3).n
		ow=0
		dw=modelFloat.getHyper().getAxis(3).d
		nt = 2*(nw-1)
		ot=0
		dt = 1./((nw-1)*dw)
		timeAxis=Hypercube.axis(n=nt,o=ot,d=dt)
		dataHyper=Hypercube.hypercube(axes=[modelFloat.getHyper().getAxis(1),modelFloat.getHyper().getAxis(2),timeAxis])

		dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")
		dataFloat.scale(0.0)

		############################# Initialization ###############################
		# fft_wfld init
		if(pyinfo): print("--------------------------- FFT Wfld init --------------------------------")
		fft_wfld_op = fft_wfld.fft_wfld(modelFloat,dataFloat)

		print("*** domain and range checks *** ")
		print("* Fp - d * ")
		print("F domain: ", fft_wfld_op.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("F range: ", fft_wfld_op.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			fft_wfld_op.dotTest(1)

		#run forward
		fft_wfld_op.forward(False,modelFloat,dataFloat)

		#write data to disk
		genericIO.defaultIO.writeVector(dataFile,dataFloat)


	else:
		print("-------------------------------------------------------------------")
		print("--------- Running Python fft transform of wfld. Transform from time to freq -----------")
		print("-------------------------------------------------------------------\n")

		# Data
		dataFloat=genericIO.defaultIO.getVector(dataFile)

		# Time and freq Axes
		nt=dataFloat.getHyper().getAxis(3).n
		ot=0
		dt=dataFloat.getHyper().getAxis(3).d
		#odd
		if(nt%2 != 0):
			nw = int(nt/2+1)
		else:
			nw = int((nt+1)/2)
		ow=0
		dw = 1./((nt-1)*dt)
		freqAxis=Hypercube.axis(n=nw,o=ow,d=dw)
		modelHyper=Hypercube.hypercube(axes=[dataFloat.getHyper().getAxis(1),dataFloat.getHyper().getAxis(2),freqAxis])

		modelFloat=SepVector.getSepVector(modelHyper,storage="dataComplex")
		modelFloat.zero()

		############################# Initialization ###############################
		# fft_wfld init
		if(pyinfo): print("--------------------------- fft_wfld init --------------------------------")
		fft_wfld_op = fft_wfld.fft_wfld(modelFloat,dataFloat)

		print("*** domain and range checks *** ")
		print("* Fp - d * ")
		print("F domain: ", fft_wfld_op.getDomain().getNdArray().shape)
		print("p shape: ", modelFloat.getNdArray().shape)
		print("F range: ", fft_wfld_op.getRange().getNdArray().shape)
		print("d shape: ", dataFloat.getNdArray().shape)
		################################ DP Test ###################################
		if (parObject.getInt("dp",0)==1):
			print("\nData op dp test:")
			fft_wfld_op.dotTest(1)

		#run adjoint
		fft_wfld_op.adjoint(False,modelFloat,dataFloat)

		#write model to disk
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

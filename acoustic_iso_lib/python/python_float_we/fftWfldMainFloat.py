#!/usr/bin/env python3
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

	#get axis
	SEPaxisToFFT = parObject.getInt("axis",0)
	# Forward
	if (parObject.getInt("adj",0) == 0):
		print("-------------------------------------------------------------------")
		print("--------- Running Python fft transform of wfld. Transform from freq to time  -----------")
		print("-------------------------------------------------------------------\n")

		# Read  model
		modelFloat=genericIO.defaultIO.getVector(modelFile,storage="complex")
		ndims = modelFloat.getHyper().getNdim()
		pythonAxisToFFT = (ndims-SEPaxisToFFT)
		print("SEPaxisToFFT: "+str(SEPaxisToFFT))
		print("pythonAxisToFFT: "+str(pythonAxisToFFT))
		# Time and freq Axes
		nw=modelFloat.getHyper().getAxis(SEPaxisToFFT).n
		ow=0
		dw=modelFloat.getHyper().getAxis(SEPaxisToFFT).d
		nt = 2*(nw)-1
		ot=0
		dt = 1./((nt-1)*dw)
		timeAxis=Hypercube.axis(n=nt,o=ot,d=dt)

		axes=[]
		for iaxis in np.arange(ndims):
			if(iaxis+1==SEPaxisToFFT):
				axes.append(timeAxis)
			else:
				axes.append(modelFloat.getHyper().getAxis(iaxis+1))
		dataHyper=Hypercube.hypercube(axes=axes)

		dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")
		dataFloat.scale(0.0)

		############################# Initialization ###############################
		# fft_wfld init
		if(pyinfo): print("--------------------------- FFT Wfld init --------------------------------")
		fft_wfld_op = fft_wfld.fft_wfld(modelFloat,dataFloat,pythonAxisToFFT)

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
		ndims = dataFloat.getHyper().getNdim()
		pythonAxisToFFT = (ndims-SEPaxisToFFT)
		print("SEPaxisToFFT: "+str(SEPaxisToFFT))
		print("pythonAxisToFFT: "+str(pythonAxisToFFT))

		# Time and freq Axes
		nts=dataFloat.getHyper().getAxis(SEPaxisToFFT).n
		ots=dataFloat.getHyper().getAxis(SEPaxisToFFT).o
		dts=dataFloat.getHyper().getAxis(SEPaxisToFFT).d

		freq_np_axis = np.fft.rfftfreq(nts,dts)
		df = freq_np_axis[1]-freq_np_axis[0]
		nf= freq_np_axis.size
		freqAxis=Hypercube.axis(n=nf,o=ots,d=df)

		axes=[]
		for iaxis in np.arange(ndims):
			if(iaxis+1==SEPaxisToFFT):
				axes.append(freqAxis)
			else:
				axes.append(dataFloat.getHyper().getAxis(iaxis+1))
		modelHyper=Hypercube.hypercube(axes=axes)

		modelFloat=SepVector.getSepVector(modelHyper,storage="dataComplex")
		modelFloat.zero()

		############################# Initialization ###############################
		# fft_wfld init
		if(pyinfo): print("--------------------------- fft_wfld init --------------------------------")
		fft_wfld_op = fft_wfld.fft_wfld(modelFloat,dataFloat,pythonAxisToFFT)

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
		print(modelFloat.getNdArray().dtype)
		genericIO.defaultIO.writeVector(modelFile,modelFloat)

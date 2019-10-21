#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os
import math
import os.path

# Modeling operators
import Acoustic_iso_float

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyLCGsolver_timer as LCG_timer
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilFloat
import TpowWfld
import Mask3d
import Mask2d
import spatialDerivModule
import SecondDeriv


if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)

	print("-------------------------------------------------------------------")
	print("------------------ Full Waveform Inversion Gradient Script --------------")
	print("-------------------------------------------------------------------\n")

	############################# Initialize Operators ###############################
	#init A^(-1)(f) nonlinear op
	nl_modelFloat,nl_dataFloat,velFloat,parObject,sourcesVector,receiversVector=Acoustic_iso_float.nonlinearOpInitFloat(sys.argv)
	nonlinearOp=Acoustic_iso_float.nonlinearPropShotsGpu(nl_modelFloat,nl_dataFloat,velFloat,parObject,sourcesVector,receiversVector)
	print('nl initiated')


	# init B^(-1)(f) born op
	born_modelFloat,born_dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornOpInitFloat(sys.argv)
	bornOp=Acoustic_iso_float.BornShotsGpu(born_modelFloat,born_dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector)
	print('born initiated')

	############################ read in wavelet, velocity, and obs data ###############################
        # Read model
	nl_modelFloat=genericIO.defaultIO.getVector(parObject.getString("wavelet","noDataFile"),ndims=3)
	#nl_modelFloat_file=parObject.getString("vel","noDataFile")
	#nl_modelFloat=genericIO.defaultIO.getVector(nl_modelFloat_file)
	dataObs_file=parObject.getString("data","noDataFile")
	dataObs=genericIO.defaultIO.getVector(dataObs_file,ndims=3)

	############################ Compute fwd wfld ###############################
	nonlinearOp.forwardWavefield(0,nl_modelFloat,nl_dataFloat) # A(-1)f
	#get wfld
	nl_fwdWfld=nonlinearOp.getWfld()

	# second time deriv
	secondDerivOp = SecondDeriv.second_deriv(nl_fwdWfld,nl_fwdWfld)
	print('second deriv initiated')
	secondDerivWfld = nl_fwdWfld.clone()
	secondDerivOp.forward(0,nl_fwdWfld,secondDerivWfld) # d2p/dt2

	############################ Compute adjoint wfld ################################
	#compute adjoint data
	nl_dataFloat.scaleAdd(dataObs,1,-1)
	born_dataFloat.scaleAdd(nl_dataFloat,0,1)
	bornOp.adjointWavefield(False,born_modelFloat,born_dataFloat)

	#get wfld
	born_adjWfld=bornOp.getSecWfld()
	############################ Compute cross correlation < B*^(-1)(KA^(-1)f-d) , A^(-1)f > #####
	ccWfld = born_adjWfld.clone()
	ccWfldNdArray = ccWfld.getNdArray()
	ccWfldNdArray[:,:,:] = np.multiply(born_adjWfld.getNdArray(),secondDerivWfld.getNdArray())

	############################ Integrate cross correlation < d2p/dt2 , A(p)m-f > #####
	ccIntWfld = ccWfld.clone()
	ccIntWfld.getNdArray()[:,:,:]=ccWfld.getNdArray()[:,:,:]
	for it in np.arange(1,parObject.getInt("nts",-1)):
		ccIntWfld.getNdArray()[it,:,:] += ccIntWfld.getNdArray()[it-1,:,:]

	################################ Output Results ##################################
	prefix=parObject.getString("prefix","dummyPrefix")
	# A^(-1)f : fwd nonlinear wfld
	genericIO.defaultIO.writeVector(prefix+"_fwd_wfld.H",secondDerivWfld)

	# B*^(r-i1)(KA^(-1)f - d) : adj wfld
	genericIO.defaultIO.writeVector(prefix+"_adj_wfld.H",born_adjWfld)

	# < B*^(-1)(KA^(-1)f-d) , A^(-1)f >
	genericIO.defaultIO.writeVector(prefix+"_cc_wfld.H",ccWfld)

	# summation < B*^(-1)(KA^(-1)f-d) , A^(-1)f >
	genericIO.defaultIO.writeVector(prefix+"_ccInt_wfld.H",ccIntWfld)

#!/usr/bin/env python3
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
import Acoustic_iso_float_we
import Acoustic_iso_float_gradio

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
	print("------------------ Full Wavefield Reconstruction Inversion Gradient Script --------------")
	print("-------------------------------------------------------------------\n")

	############################# Initialize Operators ###############################
	#init A(p)
	m_modelFloat,m_dataFloat,pressureData,gradioOp= Acoustic_iso_float_gradio.gradioOpInitFloat(sys.argv)
	slsq_model_file=parObject.getString("slsq","None")
	m_modelFloat = genericIO.defaultIO.getVector(slsq_model_file)

	# second time deriv
	secondDerivOp = SecondDeriv.second_deriv(pressureData,pressureData)

	############################ Compute residual wfld ###############################
	residualWfld = pressureData.clone()
	gradioOp.forward(0,m_modelFloat,residualWfld) # A(p)m
	residualWfld.scaleAdd(m_dataFloat,1,-1) # A(p)m-f

	############################ Compute d2p/dt2 wfld ################################
	secondDerivWfld = pressureData.clone()
	secondDerivOp.forward(0,pressureData,secondDerivWfld) # d2p/dt2

	############################ Compute cross correlation < d2p/dt2 , A(p)m-f > #####
	ccWfld = pressureData.clone()
	ccWfldNdArray = ccWfld.getNdArray()
	ccWfldNdArray[:,:,:] = np.multiply(residualWfld.getNdArray(),secondDerivWfld.getNdArray())

	############################ Integrate cross correlation < d2p/dt2 , A(p)m-f > #####
	ccIntWfld = ccWfld.clone()
	ccIntWfld.getNdArray()[:,:,:]=ccWfld.getNdArray()[:,:,:]
	for it in np.arange(1,parObject.getInt("nts",-1)):
		ccIntWfld.getNdArray()[it,:,:] += ccIntWfld.getNdArray()[it-1,:,:]

	################################ Output Results ##################################
	prefix=parObject.getString("prefix","dummyPrefix")
	# second time derivative
	genericIO.defaultIO.writeVector(prefix+"_dt2_wfld.H",secondDerivWfld)

	# A(p)m - f
	genericIO.defaultIO.writeVector(prefix+"_res_wfld.H",residualWfld)

	# < d2p/dt2 , A(p)m-f >
	genericIO.defaultIO.writeVector(prefix+"_cc_wfld.H",ccWfld)

	# summation < d2p/dt2 , A(p)m-f >
	genericIO.defaultIO.writeVector(prefix+"_ccInt_wfld.H",ccIntWfld)

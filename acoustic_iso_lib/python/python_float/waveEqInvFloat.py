#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Acoustic_iso_float_we

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
import pyProblem as Prblm
import inversionUtils
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	####################### initialize genericIO classes #######################
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	pyinfo=parObject.getInt("pyinfo",1)
	epsilonEval=parObject.getInt("epsilonEval",0)
	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------ acoustic wave equation inversion --------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ acoustic wave equation inversion--------------")

	############################# Initialization ###############################
	# Wave equation op init
	if(pyinfo): print("--------------------------- Wave equation op init --------------------------------")
	modelFloat,dataFloat,elasticParamFloat,parObject,waveEquationAcousticOp = Acoustic_iso_float_we.waveEquationOpInitFloat(sys.argv)

	################################ DP Test ###################################
	if (parObject.getInt("dp",0)==1):
		print("\nWave equation op dp test:")
		waveEquationAcousticOp.dotTest(1)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInit=modelFloat.clone()
		modelInit.scale(0.0)
	else:
		modelInit=genericIO.defaultIO.getVector(modelInitFile)

	priorFile=parObject.getString("prior","none")
	if(priorFile=="none"):
		# forcing term op
		if(pyinfo): print("--------------------------- forcing term op init --------------------------------")
		print("NOT IMPLEMENTED")
		exit()

	else:
		if(pyinfo): print("--------------------------- reading in provided prior --------------------------------")
		prior=genericIO.defaultIO.getVector(priorFile)

	# read in prior

	print("*** domain and range checks *** ")
	print("* Amp - f * ")
	print("Am domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
	print("p shape: ", modelInit.getNdArray().shape)
	print("Am range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
	print("f shape: ", dataFloat.getNdArray().shape)

	############################## Solver ######################################
	# Solver
	invProb=Prblm.ProblemL2Linear(modelInit,prior,waveEquationAcousticOp)
	LCGsolver=LCG(stop,logger=inv_log)
	LCGsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	if(pyinfo): print("--------------------------- Running --------------------------------")
	LCGsolver.run(invProb,verbose=True)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

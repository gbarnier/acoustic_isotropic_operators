#!/usr/bin/env python3.5
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
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilFloat
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# Bullshit stuff
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
	if(pyinfo): print("------------------ wavefield reconstruction --------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ wavefield reconstruction --------------")

	############################# Initialization ###############################
	# Data extraction
	if(pyinfo): print("--------------------------- Data extraction init --------------------------------")
	dataSamplingOp = wriUtilFloat.data_extraction_op_init(sys.argv)

	# Wave equation op init
	if(pyinfo): print("--------------------------- Wave equation op init --------------------------------")
	modelFloat,dataFloat,elasticParamFloat,parObject = Elastic_iso_float_we.waveEquationOpInitFloat(sys.argv)
	waveEquationElasticOp=Elastic_iso_float_we.waveEquationElasticGpu(modelFloat,dataFloat,elasticParamFloat,parObject)

	# forcing term op
	if(pyinfo): print("--------------------------- forcing term op init --------------------------------")
	forcingTermOp,prior = wriUtilFloat.forcing_term_op_init(sys.argv)

	# scale prior
	prior.scale(1/(elasticParamFloat.getHyper().getAxis(1).d*elasticParamFloat.getHyper().getAxis(2).d))

	################################ DP Test ###################################
	if (parObject.getInt("dp",0)==1):
		print("\nData op dp test:")
		dataSamplingOp.dotTest(1)
		print("\nModel op dp test:")
		waveEquationElasticOp.dotTest(1)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInit=modelFloat.clone()
		modelInit.scale(0.0)
	else:
		modelInit=genericIO.defaultIO.getVector(modelInitFile)

	# Data
	dataFile=parObject.getString("data")
	dataFloat=genericIO.defaultIO.getVector(dataFile)
	# print(dataFloat.norm())
	# print(prior.norm())
	# genericIO.defaultIO.writeVector("./priorTest.H",prior)
	# waveEquationElasticOp.adjoint(False,modelInit,prior)
	# genericIO.defaultIO.writeVector("./gradPriorTest.H",modelInit)
	# quit()
	# dataDouble=SepVector.getSepVector(dataSamplingOp.getRange().getHyper(),storage="dataDouble")
	# dataSMat=dataFloat.getNdArray()
	# dataDMat=dataDouble.getNdArray()
	# dataDMat[:]=dataSMat
	print("*** domain and range checks *** ")
	print("* Kp - d * ")
	print("K domain: ", dataSamplingOp.getDomain().getNdArray().shape)
	print("p shape: ", modelInit.getNdArray().shape)
	print("K range: ", dataSamplingOp.getRange().getNdArray().shape)
	print("K range axis 1 sampling: ", dataSamplingOp.getRange().getHyper().getAxis(1).d)
	print("d shape: ", dataFloat.getNdArray().shape)
	print("d axis 1 sampling: ", dataFloat.getHyper().getAxis(1).d)
	print("* Amp - f * ")
	print("Am domain: ", waveEquationElasticOp.getDomain().getNdArray().shape)
	print("p shape: ", modelInit.getNdArray().shape)
	print("Am range: ", waveEquationElasticOp.getRange().getNdArray().shape)
	print("f shape: ", prior.getNdArray().shape)

	############################# Regularization ###############################
	epsilon=parObject.getFloat("epsScale",1.0)*parObject.getFloat("eps",1.0)
	invProb=Prblm.ProblemL2LinearReg(modelInit,dataFloat,dataSamplingOp,epsilon,reg_op=waveEquationElasticOp,prior_model=prior)

	# Evaluate Epsilon
	if (epsilonEval==1):
		if(pyinfo): print("--- Epsilon evaluation ---")
		inv_log.addToLog("--- Epsilon evaluation ---")
		epsilon=invProb.estimate_epsilon(True)*parObject.getFloat("epsScale",1.0)
		invProb.epsilon=epsilon
	if(pyinfo): print("--- Epsilon value: ",epsilon," ---")
	inv_log.addToLog("--- Epsilon value: %s ---"%(epsilon))

	############################## Solver ######################################
	# Solver
	LCGsolver=LCG.LCGsolver(stop,logger=inv_log)
	LCGsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	if(pyinfo): print("--------------------------- Running --------------------------------")
	LCGsolver.run(invProb,verbose=True)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

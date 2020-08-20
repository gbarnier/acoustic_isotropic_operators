#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os
import math

# Modeling operators
import Acoustic_iso_float_we

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
import pyLCGsolver_timer as LCG_timer
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilFloat
import TpowWfld
import Mask3d
from sys_util import logger

if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
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
	# Wave equation op init
	if(pyinfo): print("--------------------------- Wave equation op init --------------------------------")
	modelFloat,_,slsqFloat,parObject,tempWaveEquationOp = Acoustic_iso_float_we.waveEquationOpInitFloat(sys.argv)
	timeMask=0;
	maskWidth=parObject.getInt("maskWidth",0)
	mask3dOp = Mask3d.mask3d(modelFloat,modelFloat,maskWidth,modelFloat.getHyper().axes[0].n-maskWidth,maskWidth,modelFloat.getHyper().axes[1].n-maskWidth,0,modelFloat.getHyper().axes[2].n-timeMask,0)
	waveEquationAcousticOp = pyOp.ChainOperator(tempWaveEquationOp,mask3dOp)

	# forcing term op
	if(pyinfo): print("--------------------------- forcing term op init ------------------------")
	fullPrior = parObject.getString("fullPrior","none")
	if(fullPrior=="none"):
		print("prior from wavelet")
		forcingTermOp,priorTmp = wriUtilFloat.forcing_term_op_init_p(sys.argv)
		prior=priorTmp.clone()
		mask3dOp.forward(0,priorTmp,prior)
	else:
		print("full prior")
		priorTmp=genericIO.defaultIO.getVector(fullPrior)
		prior=priorTmp.clone()
		mask3dOp.forward(0,priorTmp,prior)

	# Data extraction
	if(pyinfo): print("--------------------------- Data extraction init --------------------------------")
	sampleFullWfld = parObject.getInt("sampleFullWfld",0)
	if(sampleFullWfld==0):
		_,dataFloat,dataSamplingOp= wriUtilFloat.data_extraction_op_init(sys.argv)
	elif(sampleFullWfld==1):
		_,dataFloat,dataSamplingOp = wriUtilFloat.wfld_extraction_reg_op_init(sys.argv)
	#_,dataFloat,dataSamplingOp = wriUtilFloat.data_extraction_reg_op_init(sys.argv)

	################################ DP Test ###################################
	if (parObject.getInt("dp",0)==1):
		print("\nData op dp test:")
		dataSamplingOp.dotTest(1)
		print("\nModel op dp test:")
		#waveEquationAcousticOp.dotTest(1)
		tempWaveEquationOp.dotTest(1)
		#mask3dOp.dotTest(1)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInit=modelFloat.clone()
		modelInit.scale(0.0)
	else:
		modelInit=genericIO.defaultIO.getVector(modelInitFile)
        #		modelInit=modelFloat.clone()
	#	mask3dOp.forward(0,modelInitTmp,modelInit)


	# Data
	dataFile=parObject.getString("data")
	#dataFloatTmp0=genericIO.defaultIO.getVector(dataFile)
	#mask3dOp.forward(0,dataFloatTmp0,dataFloat)
	dataFloat=genericIO.defaultIO.getVector(dataFile)

	print("*** domain and range checks *** ")
	print("* Kp - d * ")
	print("K domain: ", dataSamplingOp.getDomain().getNdArray().shape)
	print("p shape: ", modelInit.getNdArray().shape)
	print("K range: ", dataSamplingOp.getRange().getNdArray().shape)
	print("K range axis 1 sampling: ", dataSamplingOp.getRange().getHyper().getAxis(1).d)
	print("d shape: ", dataFloat.getNdArray().shape)
	print("d axis 1 sampling: ", dataFloat.getHyper().getAxis(1).d)
	print("* Amp - f * ")
	print("Am domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
	print("p shape: ", modelInit.getNdArray().shape)
	print("Am range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
	print("f shape: ", prior.getNdArray().shape)

	############################# Regularization ###############################
	# Evaluate Epsilon
	if (epsilonEval==1):
		if(pyinfo): print("--- Epsilon evaluation ---")
		inv_log.addToLog("--- Epsilon evaluation ---")
		#epsilon=invProb.estimate_epsilon(True)*parObject.getFloat("epsScale",1.0)
		#invProb.epsilon=epsilon

		#make first data residual
		K_resid = SepVector.getSepVector(dataSamplingOp.getRange().getHyper(),storage="dataFloat")
		K_resid.scaleAdd(dataFloat,0,-1)
		dataSamplingOp.forward(1,modelInit,K_resid)

		#make first model residual
		A_resid = SepVector.getSepVector(waveEquationAcousticOp.getRange().getHyper(),storage="dataFloat")
		A_resid.scaleAdd(prior,0,-1)
		waveEquationAcousticOp.forward(1,modelInit,A_resid)

		if(modelInitFile=="None"):
			#update model
			modelOne = SepVector.getSepVector(modelInit.getHyper(),storage="dataFloat")
			modelOne.scale(0.0)
			dataSamplingOp.adjoint(1,modelOne,K_resid)
			waveEquationAcousticOp.adjoint(1,modelOne,A_resid)
			dataSamplingOp.forward(1,modelOne,K_resid)
			waveEquationAcousticOp.forward(1,modelOne,A_resid)

		epsilon = parObject.getFloat("epsScale",1.0)*math.sqrt(K_resid.dot(K_resid)/A_resid.dot(A_resid))
	else:
		epsilon=parObject.getFloat("epsScale",1.0)*parObject.getFloat("eps",1.0)


	if(pyinfo): print("--- Epsilon value: ",epsilon," ---")
	inv_log.addToLog("--- Epsilon value: %s ---"%(epsilon))


	############################# Preconditioning ###############################
	tpow=parObject.getFloat("tpowPrecond",0.0)
	gf=parObject.getInt("gfPrecond",0)
	if(tpow != 0.0):
		precondStart=parObject.getFloat("precondStart",0.0)
		if(pyinfo): print("--- Preconditioning w/ tpow: ",tpow," ---")
		tpowOp = TpowWfld.tpow_wfld(modelFloat,modelFloat,tpow,precondStart)
		invProb=Prblm.ProblemL2LinearReg(modelInit,dataFloat,dataSamplingOp,epsilon,reg_op=waveEquationAcousticOp,prior_model=prior,prec=tpowOp)
		testtpmodel = modelFloat.clone()
		testtpdata = modelFloat.clone()
		testtpmodel.set(1)
		tpowOp.forward(0,testtpmodel,testtpdata)
		genericIO.defaultIO.writeVector("tpTest.H",testtpdata)
	elif(gf == 1):
		precondStart=parObject.getFloat("precondStart",0.0)
		if(pyinfo): print("--- Preconditioning w/ greens function ---")
		_,_,gfOp= wriUtilFloat.greens_function_op_init(sys.argv)
		invProb=Prblm.ProblemL2LinearReg(modelInit,dataFloat,dataSamplingOp,epsilon,reg_op=waveEquationAcousticOp,prior_model=prior,prec=gfOp)
		testgfmodel = modelFloat.clone()
		testgfdata = modelFloat.clone()
		testgfmodel.set(1)
		gfOp.forward(0,testgfmodel,testgfdata)
		genericIO.defaultIO.writeVector("gfTest.H",testgfdata)
	else:
		if(pyinfo): print("--- No preconditioning ---")
		invProb=Prblm.ProblemL2LinearReg(modelInit,dataFloat,dataSamplingOp,epsilon,reg_op=waveEquationAcousticOp,prior_model=prior)

	############################## Solver ######################################
	# Solver
	LCGsolver=LCG(stop,logger=inv_log)
	#LCGsolver=LCG_timer.LCGsolver(stop,logger=inv_log)
	LCGsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	if(pyinfo): print("--------------------------- Running --------------------------------")
	LCGsolver.run(invProb,verbose=True)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

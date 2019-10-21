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
import wriUtilFloat
import SampleWfld
import Mask3d
import TpowWfld

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
from sys_util import logger

# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	####################### initialize genericIO classes #######################
	parObject=genericIO.io(params=sys.argv)
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
	modelFloat,dataFloat,slsqFloat,parObject,tempWaveEquationOp = Acoustic_iso_float_we.waveEquationOpInitFloat(sys.argv)
	timeMask=0;
	maskWidth=parObject.getInt("maskWidth",0)
	mask3dOp = Mask3d.mask3d(modelFloat,modelFloat,maskWidth,modelFloat.getHyper().axes[0].n-maskWidth,maskWidth,modelFloat.getHyper().axes[1].n-maskWidth,0,modelFloat.getHyper().axes[2].n-timeMask,0)
	waveEquationAcousticOp = pyOp.ChainOperator(tempWaveEquationOp,mask3dOp)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInit=modelFloat.clone()
		modelInit.scale(0.0)
	else:
		modelInit=genericIO.defaultIO.getVector(modelInitFile)

	# forcing term op
	if(pyinfo): print("--------------------------- forcing term op init ------------------------")
	fullPrior = parObject.getString("fullPrior","none")
	if(fullPrior=="none"):
		print("prior from wavelet")
		forcingTermOp,priorTmp = wriUtilFloat.forcing_term_op_init_p(sys.argv)
		prior=priorTmp.clone()
		prior.scale(0.0)
		mask3dOp.forward(0,priorTmp,prior)

	else:
		print("full prior")
		prior=genericIO.defaultIO.getVector(fullPrior)


	print("*** domain and range checks *** ")
	print("* Amp - f * ")
	print("Am domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
	print("p shape: ", modelInit.getNdArray().shape)
	print("Am range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
	print("f shape: ", dataFloat.getNdArray().shape)

	################################ DP Test ###################################
	if (parObject.getInt("dp",0)==1):
		if(pyinfo): print("--------------------------- Performing DP Test --------------------------------")
		print("\nWave equation op dp test:")
		tempWaveEquationOp.dotTest(1)
		#print("\nWfld time samping op dp test:")
		#timeSamplingOp.dotTest(1)
	if(parObject.getInt("evalConditionNumber",0)==1):
		if(pyinfo): print("--------------------------- Evaluating Condition Number --------------------------------")
		eigen = waveEquationAcousticOp.powerMethod(verbose=True,n_iter=500,eval_min=True)
		print("Max eigenvalue = %s"%(eigen[0]))
		print("Min eigenvalue = %s"%(eigen[1]))
		print("Condition number = %s"%(eigen[0]/eigen[1]))
		quit()
	############################## Solver ######################################

	############################# Preconditioning ###############################
	tpow=parObject.getFloat("tpowPrecond",0.0)
	gf=parObject.getInt("gfPrecond",0)
	if(tpow != 0.0):
		precondStart=parObject.getFloat("precondStart",0.0)
		if(pyinfo): print("--- Preconditioning w/ tpow: ",tpow," ---")
		tpowOp = TpowWfld.tpow_wfld(modelFloat,dataFloat,tpow,precondStart)
		invProb=Prblm.ProblemL2Linear(modelInit,prior,waveEquationAcousticOp,prec=tpowOp)
		testtpmodel = modelFloat.clone()
		testtpdata = modelFloat.clone()
		testtpmodel.set(1)
		tpowOp.forward(0,testtpmodel,testtpdata)
		genericIO.defaultIO.writeVector("tpTest.H",testtpdata)
	elif(gf == 1):
		precondStart=parObject.getFloat("precondStart",0.0)
		if(pyinfo): print("--- Preconditioning w/ greens function ---")
		_,_,gfOp= wriUtilFloat.greens_function_op_init(sys.argv)
		invProb=Prblm.ProblemL2Linear(modelInit,prior,waveEquationAcousticOp,prec=gfOp)
		testgfmodel = modelFloat.clone()
		testgfdata = modelFloat.clone()
		testgfmodel.set(1)
		gfOp.forward(0,testgfmodel,testgfdata)
		genericIO.defaultIO.writeVector("gfTest.H",testgfdata)
	else:
		if(pyinfo): print("--- No preconditioning ---")
		invProb=Prblm.ProblemL2Linear(modelInit,prior,waveEquationAcousticOp)

	LCGsolver=LCG.LCGsolver(stop,logger=inv_log)
	LCGsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	if(pyinfo): print("--------------------------- Running --------------------------------")
	LCGsolver.run(invProb,verbose=True)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

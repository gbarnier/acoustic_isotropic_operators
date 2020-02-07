#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os
import math

# Modeling operators
import Acoustic_iso_float_we_freq

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilFloat
import TpowWfld
import Mask4d
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
	if(pyinfo): print("------------------ wavefield reconstruction freq domain (multi experiments) --------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ wavefield reconstruction freq domain (multi experiments) --------------")

	############################# Initialization ###############################
	# Wave equation op init
	if(pyinfo): print("--------------------------- Wave equation op init --------------------------------")
	modelFloat,priorFloat,slsqFloat,parObject,tempWaveEquationOp,fftOpWfld,modelFloatTime = Acoustic_iso_float_we_freq.waveEquationOpInitFloat_multi_exp_freq(sys.argv)
	# timeMask=0;
	# maskWidth=parObject.getInt("maskWidth",0)
	# mask4dOp = Mask4d.mask4d(modelFloat,modelFloat,maskWidth,modelFloat.getHyper().axes[0].n-maskWidth,maskWidth,modelFloat.getHyper().axes[1].n-maskWidth,0,modelFloat.getHyper().axes[2].n-timeMask,0,modelFloat.getHyper().axes[3].n,0)
	# waveEquationAcousticOp = pyOp.ChainOperator(tempWaveEquationOp,mask4dOp)
	waveEquationAcousticOp=tempWaveEquationOp

	# forcing term op
	if(pyinfo): print("--------------------------- forcing term op init ------------------------")
	fullPrior = parObject.getString("fullPrior","none")
	if(fullPrior=="none"):
		print("prior from wavelet")
		forcingTermOp,priorTimeTmp = wriUtilFloat.forcing_term_op_init_p_multi_exp(sys.argv)
		#priorTime=priorTimeTmp.clone()
		prior=priorFloat.clone()
		#mask4dOp.forward(0,priorTimeTmp,priorTime)
		fftOpWfld.adjoint(0,prior,priorTimeTmp) # convert to freq
		genericIO.defaultIO.writeVector('freq_prior.H',prior)
	else:
		print("full prior")
		prior=genericIO.defaultIO.getVector(fullPrior)

	# Data extraction
	if(pyinfo): print("--------------------------- Data extraction init --------------------------------")
	sampleFullWfld = parObject.getInt("sampleFullWfld",0)
	if(sampleFullWfld==0):
		_,dataFloat,dataSamplingOp,fftOpData,dataFloatTime= wriUtilFloat.data_extraction_op_init_multi_exp_freq(sys.argv)
	elif(sampleFullWfld==1):
		print('Error sampleFullWfld=1 not implimented!')
		quit()

	################################ DP Test ###################################
	if (parObject.getInt("dp",0)==1):
		print("\nData op dp test:")
		dataSamplingOp.dotTest(1)
		print("\nModel op dp test:")
		waveEquationAcousticOp.dotTest(1)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("initial_p_model","None")
	inputMode=parObject.getString("inputMode","freq")
	if (modelInitFile=="None"):
		modelFloat.scale(0.0)
		timeFloat=fftOpWfld.getRange().clone()
		timeFloat.zero()
	else:
		#check if provided in time domain
		if (inputMode == 'time'):
			print('------ input model in time domain. converting to freq ------')
			timeFloat = genericIO.defaultIO.getVector(modelInitFile)
			print("FFT domain: ", fftOpWfld.getDomain().getNdArray().shape)
			print("modelFloat: ", modelFloat.getNdArray().shape)
			print("FFT range: ", fftOpWfld.getRange().getNdArray().shape)
			print("timeFloat: ", timeFloat.getNdArray().shape)
			fftOpWfld.adjoint(0,modelFloat,timeFloat)
		else:
			modelFloat=genericIO.defaultIO.getVector(modelInitFile)
			timeFloat=fftOpWfld.getRange().clone()
			timeFloat.zero()


	# Data
	dataFile=parObject.getString("data")
	if (inputMode == 'time'):
		print('------ input data in time domain. converting to freq ------')
		dataFloatTime=genericIO.defaultIO.getVector(dataFile)
		fftOpData.adjoint(0,dataFloat,dataFloatTime)
		genericIO.defaultIO.writeVector('freq_data.H',dataFloat)
	else:
		dataFloat=genericIO.defaultIO.getVector(dataFile)
	#dataFloatTmp0=genericIO.defaultIO.getVector(dataFile)
	#mask4dOp.forward(0,dataFloatTmp0,dataFloat)
	#dataFloat=genericIO.defaultIO.getVector(dataFile)

	print("*** domain and range checks *** ")
	print("* Kp - d * ")
	print("K domain: ", dataSamplingOp.getDomain().getNdArray().shape)
	print("K domain axis 3 sampling: ", dataSamplingOp.getDomain().getHyper().getAxis(3).d)
	print("p shape: ", modelFloat.getNdArray().shape)
	print("p axis 3 sampling: ", modelFloat.getHyper().getAxis(3).d)
	print("K range: ", dataSamplingOp.getRange().getNdArray().shape)
	print("K range axis 2 sampling: ", dataSamplingOp.getRange().getHyper().getAxis(2).d)
	print("d shape: ", dataFloat.getNdArray().shape)
	print("d axis 2 sampling: ", dataFloat.getHyper().getAxis(2).d)
	print("* Amp - f * ")
	print("Am domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
	print("p shape: ", modelFloat.getNdArray().shape)
	print("Am range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
	print("f shape: ", prior.getNdArray().shape)

	############################# Regularization ###############################
	############################# Evaluate epsilon ###############################
	# Evaluate Epsilon for p inversion
	if (epsilonEval==1):
		if(pyinfo): print("--- Epsilon evaluation ---")
		inv_log.addToLog("--- Epsilon evaluation ---")
		epsilon = wriUtilFloat.evaluate_epsilon(modelFloat,dataFloat,prior,dataSamplingOp,waveEquationAcousticOp,parObject)
	else:
		epsilon=parObject.getFloat("eps_p_scale",1.0)*parObject.getFloat("eps_p",1.0)
	if(pyinfo): print("--- Epsilon value: ",epsilon," ---")
	inv_log.addToLog("--- Epsilon value: %s ---"%(epsilon))


	############################# Preconditioning ###############################
	precond=parObject.getString("precond","None")
	if(precond=='invDiag'):
		if(pyinfo): print("--- Preconditioning w/ inverse diag ---")
		precondOp=Acoustic_iso_float_we_freq.waveEquationAcousticCpu_multi_exp_freq_precond(modelFloat,prior,slsqFloat)
	else:
		if(pyinfo): print("--- No Preconditioning ---")
		precondOp=None
	# tpow=parObject.getFloat("tpowPrecond",0.0)
	# gf=parObject.getInt("gfPrecond",0)
	# if(tpow != 0.0):
	# 	precondStart=parObject.getFloat("precondStart",0.0)
	# 	if(pyinfo): print("--- Preconditioning w/ tpow: ",tpow," ---")
	# 	tpowOp = TpowWfld.tpow_wfld(modelFloat,modelFloat,tpow,precondStart)
	# 	invProb=Prblm.ProblemL2LinearReg(modelInit,dataFloat,dataSamplingOp,epsilon,reg_op=waveEquationAcousticOp,prior_model=prior,prec=tpowOp)
	# 	testtpmodel = modelFloat.clone()
	# 	testtpdata = modelFloat.clone()
	# 	testtpmodel.set(1)
	# 	tpowOp.forward(0,testtpmodel,testtpdata)
	# 	genericIO.defaultIO.writeVector("tpTest.H",testtpdata)
	# elif(gf == 1):
	# 	precondStart=parObject.getFloat("precondStart",0.0)
	# 	if(pyinfo): print("--- Preconditioning w/ greens function ---")
	# 	_,_,gfOp= wriUtilFloat.greens_function_op_init(sys.argv)
	# 	invProb=Prblm.ProblemL2LinearReg(modelInit,dataFloat,dataSamplingOp,epsilon,reg_op=waveEquationAcousticOp,prior_model=prior,prec=gfOp)
	# 	testgfmodel = modelFloat.clone()
	# 	testgfdata = modelFloat.clone()
	# 	testgfmodel.set(1)
	# 	gfOp.forward(0,testgfmodel,testgfdata)
	# 	genericIO.defaultIO.writeVector("gfTest.H",testgfdata)
	# else:
	# 	if(pyinfo): print("--- No preconditioning ---")
	# 	invProb=Prblm.ProblemL2LinearReg(modelInit,dataFloat,dataSamplingOp,epsilon,reg_op=waveEquationAcousticOp,prior_model=prior)
	invProb=Prblm.ProblemL2LinearReg(modelFloat,dataFloat,dataSamplingOp,epsilon,reg_op=waveEquationAcousticOp,prior_model=prior,prec=precondOp)

	############################## Solver ######################################
	# Solver
	LCGsolver=LCG(stop,logger=inv_log)
	#LCGsolver=LCG_timer.LCGsolver(stop,logger=inv_log)
	LCGsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	if(pyinfo): print("--------------------------- Running --------------------------------")
	LCGsolver.run(invProb,verbose=True)

	#check if output wanted in time domain
	outputMode=parObject.getString("outputMode","freq")
	if (outputMode == 'time'):
		print('------ output mode is time domain. converting to time and writing to '+prefix+'_inv_mod_time.H------')
		fftOpWfld.forward(0,invProb.get_model(),timeFloat)
		#write data to disk
		genericIO.defaultIO.writeVector(prefix+'_inv_mod_time.H',timeFloat)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

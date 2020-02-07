#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Modeling operators
import Acoustic_iso_float_we_freq
import wriUtilFloat
import SampleWfld
import Mask4d
import TpowWfld

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
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
	if(pyinfo): print("------------------ acoustic wave equation inversion with helmholtz --------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ acoustic wave equation inversion with helmholtz --------------")

	############################# Initialization ###############################
	# Wave equation op init
	if(pyinfo): print("--------------------------- Wave equation op init --------------------------------")
	modelFloat,dataFloat,slsqFloat,parObject,tempWaveEquationOp,fftOp,modelFloatTime = Acoustic_iso_float_we_freq.waveEquationOpInitFloat_multi_exp_freq(sys.argv)
	waveEquationAcousticOp = tempWaveEquationOp
	fmin=parObject.getFloat("fmin",-1)
	if(fmin==-1):
		freqMask=1
	else:
		freqMask=int(fmin/fftOp.getDomain().getHyper().getAxis(3).d)
	print("fmin: "+str(fmin)+"windowing out first "+str(freqMask)+" samples.")
	spaceMask=10
	laplacianBufferMaskOp = Mask4d.mask4d_complex(modelFloat,modelFloat,spaceMask,modelFloat.getHyper().axes[0].n-(spaceMask+1),spaceMask,modelFloat.getHyper().axes[1].n-(spaceMask+1),freqMask,modelFloat.getHyper().axes[2].n,0,modelFloat.getHyper().axes[3].n,0)
	freqLowCutMaskOp = Mask4d.mask4d_complex(modelFloat,modelFloat,0,modelFloat.getHyper().axes[0].n,0,modelFloat.getHyper().axes[1].n,freqMask,modelFloat.getHyper().axes[2].n,0,modelFloat.getHyper().axes[3].n,0)
	waveEquationAcousticOp = pyOp.ChainOperator(freqLowCutMaskOp,pyOp.ChainOperator(tempWaveEquationOp,laplacianBufferMaskOp))
	#waveEquationAcousticOp = tempWaveEquationOp

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("initial_p_model","None")
	if (modelInitFile=="None"):
		modelFloat.scale(0.0)
	else:
		#check if provided in time domain
		inputMode=parObject.getString("inputMode","freq")
		if (inputMode == 'time'):
			print('------ input model in time domain. converting to freq ------')
			timeFloat = genericIO.defaultIO.getVector(modelInitFile,ndims=4)
			modelFloatTmp=modelFloat.clone()
			fftOp.adjoint(0,modelFloatTmp,timeFloat)
			freqLowCutMaskOp.forward(0,modelFloatTmp,modelFloat)
			genericIO.defaultIO.writeVector('freq_model.H',modelFloat)
		else:
			modelFloatTmp=genericIO.defaultIO.getVector(modelInitFile,ndims=4)
			freqLowCutMaskOp.forward(0,modelFloatTmp,modelFloat)

	# forcing term op
	if(pyinfo): print("--------------------------- forcing term op init ------------------------")
	fullPrior = parObject.getString("fullPrior","none")
	if(fullPrior=="none"):
		print("prior from wavelet")
		forcingTermOp,priorTimeTmp = wriUtilFloat.forcing_term_op_init_p_multi_exp(sys.argv)
		#priorTime=priorTimeTmp.clone()
		priorFreqTmp=dataFloat.clone()
		prior=dataFloat.clone()
		#laplacianBufferMaskOp.forward(0,priorTimeTmp,priorTime)
		fftOp.adjoint(0,priorFreqTmp,priorTimeTmp) # convert to freq
		freqLowCutMaskOp.forward(0,priorFreqTmp,prior)
		prior.writeVec('freq_prior.H')
		genericIO.defaultIO.writeVector('time_prior.H',priorTimeTmp)
	else:
		print("full prior")
		prior=genericIO.defaultIO.getVector(fullPrior)


	print("*** domain and range checks *** ")
	print("* Amp - f * ")
	print("Am domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
	print("p shape: ", modelFloat.getNdArray().shape)
	print("Am range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
	print("f shape: ", dataFloat.getNdArray().shape)

	################################ DP Test ###################################
	if (parObject.getInt("dp",0)==1):
		if(pyinfo): print("--------------------------- Performing DP Test --------------------------------")
		print("\nWave equation op dp test:")
		waveEquationAcousticOp.dotTest(1)
		#print("\nFFT op dp test:")
		#fftOp.dotTest(1)
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
		invProb=Prblm.ProblemL2Linear(modelFloat,prior,waveEquationAcousticOp,prec=tpowOp)
		testtpmodel = modelFloat.clone()
		testtpdata = modelFloat.clone()
		testtpmodel.set(1)
		tpowOp.forward(0,testtpmodel,testtpdata)
		genericIO.defaultIO.writeVector("tpTest.H",testtpdata)
	elif(gf == 1):
		precondStart=parObject.getFloat("precondStart",0.0)
		if(pyinfo): print("--- Preconditioning w/ greens function ---")
		_,_,gfOp= wriUtilFloat.greens_function_op_init(sys.argv)
		invProb=Prblm.ProblemL2Linear(modelFloat,prior,waveEquationAcousticOp,prec=gfOp)
		testgfmodel = modelFloat.clone()
		testgfdata = modelFloat.clone()
		testgfmodel.set(1)
		gfOp.forward(0,testgfmodel,testgfdata)
		genericIO.defaultIO.writeVector("gfTest.H",testgfdata)
	else:
		if(pyinfo): print("--- No preconditioning ---")
		invProb=Prblm.ProblemL2Linear(modelFloat,prior,waveEquationAcousticOp)

	LCGsolver=LCG(stop,logger=inv_log)
	LCGsolver.setDefaults(save_obj=saveObj,save_res=saveRes,save_grad=saveGrad,save_model=saveModel,prefix=prefix,iter_buffer_size=bufferSize,iter_sampling=iterSampling,flush_memory=flushMemory)

	# Run solver
	if(pyinfo): print("--------------------------- Running --------------------------------")
	LCGsolver.run(invProb,verbose=True)

	#check if output wanted in time domain
	outputMode=parObject.getString("outputMode","freq")
	if (outputMode == 'time'):
		print('------ output mode is time domain. converting to time and writing to '+prefix+'_inv_mod_time.H------')
		fftOp.forward(0,invProb.get_model(),modelFloatTime)
		#write data to disk
		genericIO.defaultIO.writeVector(prefix+'_inv_mod_time.H',modelFloatTime)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

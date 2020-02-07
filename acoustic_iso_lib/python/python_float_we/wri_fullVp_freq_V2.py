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
import Acoustic_iso_float_we_freq_V2

# Solver library
import pyOperator as pyOp
from pyLinearSolver import LCGsolver as LCG
from pyLinearSolver import LSQRsolver as LSQR
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilFloat
import TpowWfld
import Mask4d
import Mask3d
import Mask2d
import spatialDerivModule
from sys_util import logger

def printAndLog(message,pyinfo,inv_log):
	if(pyinfo): print(message)
	inv_log.addToLog(message)


if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)
	epsilonEval=parObject.getInt("epsilonEval",0)
	# Initialize parameters for inversion
	nIter,stop_m,stop_p,logFile,saveObj_p,saveRes_p,saveGrad_p,saveModel_p,saveObj_m,saveRes_m,saveGrad_m,saveModel_m,prefix,bufferSize_p,iterSampling_p,bufferSize_m,iterSampling_m,restartFolder,flushMemory,info=inversionUtils.inversionFullWRIInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	printAndLog("-------------------------------------------------------------------\n" +\
			    "-------------- Full Wavefield Reconstruction Inversion ------------\n" +\
			    "-------------------------------------------------------------------\n\n" \
				,pyinfo,inv_log)


	############################# Determine Domains ###############################
	inputMode=parObject.getString("inputMode","time")
	inversionMode=parObject.getString("inversionMode","time")
	outputMode=parObject.getString("outputMode","time")

	printAndLog("---------------------- I/O and Inversion Domains ------------------\n" +\
			    "Input domain: " + str(inputMode) + ". Which includes:\n" +\
				"\t-Initial wavefield (initial_p_model)\n" +\
				"\t-Observed data (data)\n" +\
				"\t-Forcing term (wavelet)\n" +\
				"Inversion domain: " + str(inversionMode) + ".\n" +\
				"\t-The wavefield will be in this domain. Therefore the wave equation operations\n" +\
				"\twill be done calculated in this domain. The outputed wavefield residuals,\n" +\
				"\tgradients, and models will be in this domain.\n" +\
				"Ouput domain: " + str(outputMode) + ".\n" +\
			    "-------------------------------------------------------------------\n\n" \
				,pyinfo,inv_log)

	############################# Initialize Operators ###############################
	printAndLog("----------------------- Initializing Operators --------------------\n" +\
				"Initializing..." \
				,pyinfo,inv_log)
	# Wave equation w.r.t. p op init #############################
	printAndLog("\tStarted wave equation op as function of wavefield",pyinfo,inv_log)
	if(inversionMode=="time"):
		p_model,priorFloat,slsqFloat,parObject,waveEquationAcousticOp = Acoustic_iso_float_we.waveEquationOpInitFloat_multi_exp(sys.argv)
		# timeMask=0;
		# maskWidth=parObject.getInt("maskWidth",0)
		# mask4dOp = Mask4d.mask4d(modelFloat,modelFloat,maskWidth,modelFloat.getHyper().axes[0].n-maskWidth,maskWidth,modelFloat.getHyper().axes[1].n-maskWidth,0,modelFloat.getHyper().axes[2].n-timeMask,0,modelFloat.getHyper().axes[3].n,0)
		# waveEquationAcousticOp = pyOp.ChainOperator(tempWaveEquationOp,mask4dOp)
	else:
		p_model,priorFloat,slsqFloat,parObject,waveEquationAcousticOpTemp,_,modelFloatTime = Acoustic_iso_float_we_freq_V2.waveEquationOpInitFloat_multi_exp_freq_V2(sys.argv)
		fmin=parObject.getFloat("fmin",-1)
		if(fmin==-1):
			freqMask=1
		else:
			freqMask=int(fmin/p_model.getHyper().getAxis(3).d)
		printAndLog("\tfmin: "+str(fmin)+"windowing out first "+str(freqMask)+" samples.", pyinfo,inv_log)
		spaceMask=10
		laplacianBufferMaskOp = Mask4d.mask4d_complex(p_model,p_model,spaceMask,p_model.getHyper().axes[0].n-(spaceMask+1),spaceMask,p_model.getHyper().axes[1].n-(spaceMask+1),freqMask,p_model.getHyper().axes[2].n,0,p_model.getHyper().axes[3].n,0)
		freqLowCutMaskOp = Mask4d.mask4d_complex(p_model,p_model,0,p_model.getHyper().axes[0].n,0,p_model.getHyper().axes[1].n,freqMask,p_model.getHyper().axes[2].n-3,0,p_model.getHyper().axes[3].n,0)
		waveEquationAcousticOp = pyOp.ChainOperator(freqLowCutMaskOp,pyOp.ChainOperator(waveEquationAcousticOpTemp,laplacianBufferMaskOp))
		#waveEquationAcousticOp = waveEquationAcousticOpTemp
	printAndLog("\tFinished wave equation op as function of wavefield\n",pyinfo,inv_log)

	# FFT operator #############################
	printAndLog("\tStarted FFT",pyinfo,inv_log)
	p_modelFloat_fft,_,fftOpWfldTmp = wriUtilFloat.fft_wfld_multi_exp_init(sys.argv)
	printAndLog("\tFinished FFT\n",pyinfo,inv_log)
	if(inversionMode=="time"):
		fftOpWfld=fftOpWfldTmp
	else:
		fftOpWfld=pyOp.ChainOperator(freqLowCutMaskOp,fftOpWfldTmp)

	# Data sampling operator K #############################
	printAndLog("\tStarted data sampling op",pyinfo,inv_log)
	if(inversionMode=="time"):
		_,p_data,dataSamplingOp=wriUtilFloat.data_extraction_op_init_multi_exp(sys.argv)
	else:
		_,p_data,dataSamplingOpTmp,fftOpDataTmp,dataFloatTime=wriUtilFloat.data_extraction_op_init_multi_exp_freq(sys.argv)
		freqLowCutMaskDataOp= Mask2d.mask2d_complex(fftOpDataTmp.getDomain(),fftOpDataTmp.getDomain(),0,fftOpDataTmp.getDomain().getHyper().axes[0].n,freqMask,fftOpDataTmp.getDomain().getHyper().axes[1].n-3,0)
		fftOpData=pyOp.ChainOperator(freqLowCutMaskDataOp,fftOpDataTmp)
		dataSamplingOp = pyOp.ChainOperator(freqLowCutMaskOp,dataSamplingOpTmp)
	printAndLog("\tFinished data sampling op\n",pyinfo,inv_log)

	#init forcing term operator to create f #############################
	printAndLog("\tStarted forcing term op",pyinfo,inv_log)
	forcingTermOp,priorTmp = wriUtilFloat.forcing_term_op_init_p_multi_exp(sys.argv)
	prior=priorFloat.clone()
	if(inversionMode=="time" and inputMode == 'freq'):
		print('\t\tInput wavelet in freq domain. Converting to time.')
		fftOpWfld.forward(0,priorTmp,prior)
	elif(inversionMode=="freq" and inputMode == 'time'):
		print('\t\tInput wavelet in time domain. Converting to freq.')
		fftOpWfld.adjoint(0,prior,priorTmp)
	printAndLog("\tFinished forcing term op\n",pyinfo,inv_log)

	# Wave equation w.r.t. m op init #############################
	printAndLog("\tStarted wave equation op as function of slsq",pyinfo,inv_log)
	if(inversionMode=="time"):
		m_model,m_data,pressureData,gradioOp= Acoustic_iso_float_gradio.gradioOpInitFloat_multi_exp(sys.argv)
	else:
		m_model,m_data,pressureData,gradioOp,fftOp= Acoustic_iso_float_gradio.gradioOpInitFloat_multi_exp_freq(sys.argv)
	printAndLog("\tFinished wave equation op as function of slsq\n",pyinfo,inv_log)

	# #############################
	if(pyinfo):
		print("*** domain and range checks *** ")
		print("** O(p)=||Kp-d||+eps||A(m)p-f|| **")
		print("K domain: ", dataSamplingOp.getDomain().getNdArray().shape)
		print("p shape: ", p_model.getNdArray().shape)
		print("K range: ", dataSamplingOp.getRange().getNdArray().shape)
		print("d shape: ", p_data.getNdArray().shape)
		print("A(m) domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
		print("p shape: ", p_model.getNdArray().shape)
		print("A(m) range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
		print("f shape: ", prior.getNdArray().shape)

		print("\n** A(p)m - f' ** ")
		print("A(p) domain: ", gradioOp.getDomain().getNdArray().shape)
		print("m shape: ", m_model.getNdArray().shape)
		print("A(p) range: ", gradioOp.getRange().getNdArray().shape)
		print("f' shape: ", m_data.getNdArray().shape)

	printAndLog("-------------------------------------------------------------------\n\n" \
				,pyinfo,inv_log)
	############################# Create LCG Solvers ###############################
	# p Solver
	solverType=parObject.getString("solver","LCG")
	if(solverType=="LCG"):
		p_solver=LCG(stop_p,logger=inv_log)
	elif(solverType=="LSQR"):
		p_solver=LSQR(stop_p,logger=inv_log)
	p_prefix=prefix+"_pinv"

	# p Solver
	m_LCGsolver=LCG(stop_m,logger=inv_log)
	m_prefix=prefix+"_minv"

	############################# Initial Models ###############################
	printAndLog("------------------------ Reading Initial Models -------------------",pyinfo,inv_log)

	# intial wfld #############################
	initial_p_model_file=parObject.getString("initial_p_model","None")
	if (initial_p_model_file=="None"):
		printAndLog("No initial wavefield provided... using zero wavefield",pyinfo,inv_log)
		current_p_model=p_model.clone()
		current_p_model.zero()
	else:
		printAndLog("Reading initial wavefield(initial_p_model) from "+initial_p_model_file,pyinfo,inv_log)
		if(inputMode == 'time'):
			current_p_model_time=genericIO.defaultIO.getVector(initial_p_model_file,storage='float')
			if(inversionMode=='freq'):
				printAndLog("\tInput wavefield in time domain. Converting initial wavefield to freq",pyinfo,inv_log)
				current_p_model=p_model.clone()
				fftOpWfld.adjoint(0,current_p_model,current_p_model_time)
			else:
				current_p_model=current_p_model_time
		else:
			current_p_model_freq=genericIO.defaultIO.getVector(initial_p_model_file,storage='complex')
			current_p_model=current_p_model_freq.clone()
			if(inversionMode=='freq'):
				printAndLog("\tInput wavefield in freq domain. Converting initial wavefield to time",pyinfo,inv_log)
				current_p_model=p_model.clone()
				fftOpWfld.forward(0,current_p_model_freq,current_p_model)
			else:
				current_p_model=current_p_model_freq

	initial_m_model_file=parObject.getString("initial_m_model","None")

	#initial slsq #############################
	if (initial_m_model_file=="None"):
		printAndLog("\tNo initial slsq provided... using water slowness",pyinfo,inv_log)
		if(pyinfo): print("No initial slsq provided... using water slowness")
		current_m_model=m_modelFloat.cloneSpace()
		current_m_model.set(1.0/(1500*1500))
	else:
		printAndLog("Reading initial slsq_model (initial_m_model) from " + initial_m_model_file,pyinfo,inv_log)
		current_m_model=genericIO.defaultIO.getVector(initial_m_model_file)

	printAndLog("-------------------------------------------------------------------\n\n" \
			,pyinfo,inv_log)
	############################# Observed Data ###############################
	printAndLog("------------------------ Reading Observed Data -------------------",pyinfo,inv_log)
	p_data_file=parObject.getString("data")
	printAndLog("Reading observed data (data) from "+p_data_file,pyinfo,inv_log)

	if(inputMode == 'time'):
		current_p_data_time=genericIO.defaultIO.getVector(p_data_file,storage='float')
		if(inversionMode=='freq'):
			printAndLog("\tInput data in time domain. Converting initial wavefield to freq",pyinfo,inv_log)
			current_p_data=p_data.clone()
			print(fftOpData.getDomain().getNdArray().shape)
			print(current_p_data.getNdArray().shape)
			print(fftOpData.getRange().getNdArray().shape)
			print(current_p_data_time.getNdArray().shape)
			fftOpData.adjoint(0,current_p_data,current_p_data_time)
			genericIO.defaultIO.writeVector(prefix+'_current_p_data.H',current_p_data)
		else:
			current_p_data=current_p_data_time
	else:
		current_p_data_freq=genericIO.defaultIO.getVector(p_data_file,storage='complex')
		current_p_data=current_p_data_freq.clone()
		if(inversionMode=='time'):
			printAndLog("\tInput data in freq domain. Converting initial wavefield to time",pyinfo,inv_log)
			current_p_data=p_data.clone()
			fftOpData.forward(0,current_p_data_freq,current_p_data)
		else:
			current_p_data=current_p_data_freq

	printAndLog("-------------------------------------------------------------------\n\n" \
			,pyinfo,inv_log)
	################################ DP Test ###################################
	if (parObject.getInt("dp",0)==1):
		if(pyinfo): print("\n------------------------- DP Tests ------------------------------")
		print("A(m)p dot product test")
		waveEquationAcousticOp.dotTest(1)
		print("\nK dot product test")
		dataSamplingOp.dotTest(1)
		print("\nA(p)m dot product test")
		if(current_p_model.norm() == 0):
			print("\nCurrent p model is zero so we will do DP test with rand p")
			rand_p_model = current_p_model.clone()
			rand_p_model.rand()
			gradioOp.args[1].update_wfld(rand_p_model)
			gradioOp.dotTest(1)
			gradioOp.args[1].update_wfld(current_p_model) # updates d2p/dt2
		else:
			gradioOp.args[1].update_wfld(current_p_model) # updates d2p/dt2
			gradioOp.dotTest(1)
	if(parObject.getInt("evalConditionNumber",0)==1):
		if(pyinfo): print("--------------------------- Evaluating Condition Number --------------------------------")
		test=waveEquationAcousticOp.H*waveEquationAcousticOp
		eigen = test.powerMethod(verbose=True,niter=500,eval_min=True)
		print("Max eigenvalue = %s"%(eigen[0]))
		print("Min eigenvalue = %s"%(eigen[1]))
		print("Condition number = %s"%(eigen[0]/eigen[1]))
		quit()
		############################# Regularization ###############################
	# regularization of m inversion
	minBound=parObject.getFloat("minBound", -100)
	maxBound=parObject.getFloat("maxBound", 100)
	minBoundVector=None
	maxBoundVector=None
	if(minBound!=-100):
		minBoundVector = current_m_model.clone()
		minBoundVector.set(minBound)
	if(maxBound!=100):
		maxBoundVector = current_m_model.clone()
		maxBoundVector.set(maxBound)

	# regOp_m_type=parObject.getString("regOp_m","None")
	# if (regOp_m_type!="None"):
	# 	# Get epsilon value from user
	# 	epsilon_m=parObject.getFloat("eps_m_scale",1.0)*parObject.getFloat("eps_m",1.0)
	# 	inv_log.addToLog("--- Epsilon value: %s ---"%(epsilon_m))
	# 	maskWidth=parObject.getInt("maskWidth",0)
	# 	mask2dOp = Mask2d.mask2d(m_modelFloat,m_modelFloat,maskWidth,m_modelFloat.getHyper().axes[0].n-maskWidth,maskWidth,m_modelFloat.getHyper().axes[1].n-maskWidth,0)
	#
	# 	# Spatial gradient in z-direction
	# 	if (regOp_m_type=="zGrad"):
	# 		if(pyinfo): print("--- Vertical gradient regularization ---")
	# 		inv_log.addToLog("--- Vertical gradient regularization ---")
	# 		fat=spatialDerivModule.zGradInit(sys.argv)
	# 		tempGradOp=spatialDerivModule.zGradPython(current_m_model,current_m_model,fat)
	# 		regOp_m = pyOp.ChainOperator(tempGradOp,mask2dOp)
	#
	# 	# Spatial gradient in x-direction
	# 	if (regOp_m_type=="xGrad"):
	# 		if(pyinfo): print("--- Horizontal gradient regularization ---")
	# 		inv_log.addToLog("--- Horizontal gradient regularization ---")
	# 		fat=spatialDerivModule.xGradInit(sys.argv)
	# 		tempGradOp=spatialDerivModule.xGradPython(current_m_model,current_m_model,fat)
	# 		regOp_m = pyOp.ChainOperator(tempGradOp,mask2dOp)
	#
	# 	# Sum of spatial gradients in z and x-directions
	# 	if (regOp_m_type=="zxGrad"):
	# 		if(pyinfo): print("--- Gradient regularization in both directions ---")
	# 		inv_log.addToLog("--- Gradient regularization in both directions ---")
	# 		fat=spatialDerivModule.zxGradInit(sys.argv)
	# 		tempGradOp=spatialDerivModule.zxGradPython(current_m_model,current_m_model,fat)
	# 		regOp_m = pyOp.ChainOperator(tempGradOp,mask2dOp)
	# else:
	# 	epsilon_m=0
	# 	regOp_m=None
	# 	if(pyinfo): print("--- No regularization of m inversion ---")
	regOp_m=None
	epsilon_m=0
	############################# Gradient Editing ##############################
	# gradEdit = parObject.getInt("gradEdit",0)
	# if(gradEdit==1):
	# 	gradEditOp=wriUtilFloat.grad_edit_mora
	# elif(gradEdit==2):
	# 	gradEditOp=wriUtilFloat.grad_edit_diving
	# else:
	# 	gradEditOp=None
	gradEditOp=None
	############################# Preconditioning ###############################
	# p inv
	# precond_p=parObject.getString("precondOp_p","None")
	# if(precond_p == "tpow"):
	# 	precondStart=parObject.getFloat("precondStart",0.0)
	# 	precondTpow=parObject.getFloat("precondTpow",1.0)
	# 	if(pyinfo): print("--- Preconditioning p inversion w/ tpow: ",precondTpow," ---")
	# 	precondOp_p = TpowWfld.tpow_wfld(modelFloat,modelFloat,precondTpow,precondStart)
	# elif(precond_p == "gf"):
	# 	if(pyinfo): print("--- Preconditioning p inversion w/ greens function ---")
	# 	_,_,precondOp_p= wriUtilFloat.greens_function_op_init(sys.argv)
	# else:
	# 	if(pyinfo): print("--- No preconditioning of p inversion ---")
	# 	inv_log.addToLog("--- No preconditioning of p inversion ---")
	# 	precondOp_p = None
	#
	# # m inv
	# precond_m=parObject.getString("precondOp_m","None")
	# if(precond_m == "stack"):
	# 	if(pyinfo): print("--- Preconditioning w/ wfld stack ---")
	# 	inv_log.addToLog("--- Preconditioning w/ wfld stack ---")
	# 	precondOp_m = SphericalSpreadingScale.spherical_spreading_scale_wfld(modelInit,modelInit,pressureData)
	# else:
	# 	if(pyinfo): print("--- No preconditioning of m inversion---")
	# 	inv_log.addToLog("--- No preconditioning of m inversion ---")
	# 	precondOp_m = None
	precondOp_m = None
	precondOp_p = None
	############################# Begin Inversion ###############################
	printAndLog("-------------- Checking restart parameters -----------------------\n" \
			,pyinfo,inv_log)

	# check restart
	restartIter=parObject.getInt("restartIter",-1)
	if(restartIter==-1): #no restart
		pFinished=parObject.getInt("pFinished",0)
		restartIter=0
	else: # restarting from restartIter
		pFinished=parObject.getInt("pFinished",0)
		if(pFinished==1): # wavefield recon is done for iter restartIter
			initial_p_model_file=p_prefix+"_iter"+str(restartIter)+"_inv_mod.H"
			current_p_model=genericIO.defaultIO.getVector(initial_p_model_file)
		else: # wavefield recon is NOT done for iter restartIter
			initial_p_model_file=p_prefix+"_iter"+str(restartIter-1)+"_inv_mod.H"
			current_p_model=genericIO.defaultIO.getVector(initial_p_model_file)
		if(restartIter!=0):
			initial_m_model_file = m_prefix+"_iter"+str(restartIter-1)+"_inv_mod.H"
			current_m_model=genericIO.defaultIO.getVector(initial_m_model_file)
	printAndLog("\tStarting at iteration " + str(restartIter) \
					,pyinfo,inv_log)
	if(pFinished==0):
		 printAndLog("\n\tat wavefield reconstruction step.",pyinfo,inv_log)
	else:
		printAndLog("\n\tat gradiometry step.",pyinfo,inv_log)
	printAndLog("\n\tUsing wavefield from " + str(initial_p_model_file),pyinfo,inv_log)
	printAndLog("\n\tUsing slsq from " + str(initial_m_model_file),pyinfo,inv_log)

	printAndLog("-------------------------------------------------------------------\n\n" \
			,pyinfo,inv_log)
	############################# Evaluate epsilon ###############################
	#need to set earth model in wave equation operator
	waveEquationAcousticOp.args[0].args[1].update_slsq(current_m_model)
	#waveEquationAcousticOp.update_slsq(current_m_model)
	# Evaluate Epsilon for p inversion
	if (epsilonEval==1):
		printAndLog("------------------------ Epsilon evaluation -----------------------\n" \
				,pyinfo,inv_log)
		epsilon_p = wriUtilFloat.evaluate_epsilon(current_p_model,current_p_data,prior,dataSamplingOp,waveEquationAcousticOp,parObject)
		eps_p_scale=parObject.getFloat("eps_p_scale",1.0)
		print("\tepsilon set to: " + str(epsilon_p))
		print("\tepsilon scale: " + str(eps_p_scale))
		epsilon_p=eps_p_scale*epsilon_p
		print("\tnew epsilon value: " + str(epsilon_p))
		p_invProb=Prblm.ProblemL2LinearReg(current_p_model,current_p_data,dataSamplingOp,epsilon_p,reg_op=waveEquationAcousticOp,prior_model=prior,prec=precondOp_p)
		#estimate_epsilon
	elif(epsilonEval==2):
		# niter_pm=100
		# dataSamplingOp.powerMethod(verbose=True,niter=niter_pm)
		# waveEquationAcousticOp.powerMethod(verbose=True,niter=niter_pm)
		p_invProb=Prblm.ProblemL2LinearReg(current_p_model,current_p_data,dataSamplingOp,epsilon=1,reg_op=waveEquationAcousticOp,prior_model=prior,prec=precondOp_p)
		epsilon_p=p_invProb.estimate_epsilon(verbose=True)
	else:
		epsilon_p=parObject.getFloat("eps_p",1.0)
		eps_p_scale=parObject.getFloat("eps_p_scale",1.0)
		print("\tepsilon set to: " + str(epsilon_p))
		print("\tepsilon scale: " + str(eps_p_scale))
		epsilon_p=eps_p_scale*epsilon_p
		print("\tnew epsilon value: " + str(epsilon_p))
		p_invProb=Prblm.ProblemL2LinearReg(current_p_model,current_p_data,dataSamplingOp,epsilon=epsilon_p,reg_op=waveEquationAcousticOp,prior_model=prior,prec=precondOp_p)
	#printAndLog("\tEpsilon value: "+str(epsilon_p),pyinfo,inv_log)
	printAndLog("-------------------------------------------------------------------\n\n" \
		,pyinfo,inv_log)
	# for number of outer loops
	for iteration in range(restartIter,nIter):
		printAndLog("------------------- Running Iteration "+str(iteration)+" ------------------------\n" \
				,pyinfo,inv_log)

		if(pFinished==0):
			# minimize ||Kp-d||+e^2/2||A(m)p-f|| w.r.t. p
			# update m
			waveEquationAcousticOp.args[0].args[1].update_slsq(current_m_model)
			#waveEquationAcousticOp.update_slsq(current_m_model)
			#re-evaluate epsilon
			# if(parObject.getInt("reEvalEpsilon",0)!=0):
			# 	epsilon_p = wriUtilFloat.evaluate_epsilon(current_p_model,current_p_data,prior,dataSamplingOp,waveEquationAcousticOp,parObject)
			# 	printAndLog("\tRe-evalute epsilon to: "+str(epsilon_p)+" ----------------------\n" \
			# 			,pyinfo,inv_log)
			# elif(parObject.getFloat("reScaleEpsilon",0)!=0 and iteration!=0 ):
			# 	scale = parObject.getFloat("reScaleEpsilon",0)
			# 	epsilon_p = epsilon_p * scale
			# 	printAndLog("Re-scale epsilon by "+str(scale)+". Epsiilon is now: "+str(epsilon_p) \
			# 			,pyinfo,inv_log)
			# reinit L2 problem with new initial p
			# p_invProb=Prblm.ProblemL2LinearReg(current_p_model,current_p_data,dataSamplingOp,epsilon_p,reg_op=waveEquationAcousticOp,prior_model=prior,prec=precondOp_p)
			# update prefix and solver defaults
			p_prefix_cur=p_prefix+"_iter"+str(iteration)
			print("p_prefix_cur: "+str(p_prefix_cur))
			p_solver.setDefaults(save_obj=saveObj_p,save_res=saveRes_p,save_grad=saveGrad_p,save_model=saveModel_p,prefix=p_prefix_cur,iter_buffer_size=bufferSize_p,iter_sampling=iterSampling_p,flush_memory=flushMemory)
			# run LCG solver
			p_solver.run(p_invProb,verbose=True)
			# update current p model
			current_p_model=p_solver.inv_model
		pFinished=0
		# minimize ||A(p)m-f|| w.r.t. m
		# update p and f
		#m_dataFloat,_ = Acoustic_iso_float_gradio.update_data(current_p_model,sys.argv) #f is actaully f+Lapl(p) so we need to update
		#gradioOp.get_op1().update_wfld(current_p_model) # updates d2p/dt2
		# create new gradioOp
		if(inversionMode=="time"):
			_,m_data,pressureData,gradioOp=Acoustic_iso_float_gradio.gradioOpInitFloat_multi_exp(sys.argv)
		else:
			_,m_data,pressureData,gradioOp,_=Acoustic_iso_float_gradio.gradioOpInitFloat_multi_exp_freq(sys.argv,pressureData=current_p_model)
		#update prec
		# reinit L2 problem with new initial m
		m_invProb=Prblm.ProblemL2LinearReg(current_m_model,m_data,gradioOp,epsilon_m,minBound=minBoundVector,maxBound=maxBoundVector,reg_op=regOp_m,prec=precondOp_m)
		# update prefix and solver defaults
		m_prefix_cur=m_prefix+"_iter"+str(iteration)
		m_LCGsolver.setDefaults(save_obj=saveObj_m,save_res=saveRes_m,save_grad=saveGrad_m,save_model=saveModel_m,prefix=m_prefix_cur,iter_buffer_size=bufferSize_m,iter_sampling=iterSampling_m,flush_memory=flushMemory)
		# run LCG solver
		m_LCGsolver.run(m_invProb,verbose=True)
		#update current m model
		current_m_model=m_LCGsolver.inv_model

	printAndLog("-------------------------------------------------------------------\n" +\
				"--------------------------- All done ------------------------------\n" +\
				"-------------------------------------------------------------------\n" \
				,pyinfo,inv_log)

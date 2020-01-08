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
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilFloat
import TpowWfld
import Mask3d
import Mask2d
import spatialDerivModule
from sys_util import logger

if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
	pyinfo=parObject.getInt("pyinfo",1)
	epsilonEval=parObject.getInt("epsilonEval",0)
	# Initialize parameters for inversion
	nIter,stop_m,stop_p,logFile,saveObj_p,saveRes_p,saveGrad_p,saveModel_p,saveObj_m,saveRes_m,saveGrad_m,saveModel_m,prefix,bufferSize_p,iterSampling_p,bufferSize_m,iterSampling_m,restartFolder,flushMemory,info=inversionUtils.inversionFullWRIInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------ Full Wavefield Reconstruction Inversion --------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ Full Wavefield Reconstruction Inversion --------------")

	############################# Initialize Operators ###############################
	#init F
	p_modelFloat_fft,_,FFTop = wriUtilFloat.fft_wfld_init(sys.argv)
	#FFTop = pyOp.Transpose(FFTop)

	#init A(m)
	_,_,slsqFloat,parObject,tempWaveEquationOp = Acoustic_iso_float_we.waveEquationOpInitFloat(sys.argv)
#	waveEquationAcousticOp = pyOp.ChainOperator(tempWaveEquationOp,mask3dOp)
	waveEquationAcousticOp = pyOp.ChainOperator(FFTop,tempWaveEquationOp)

	#init K
	_,p_dataFloat,tempDataSamplingOp= wriUtilFloat.data_extraction_op_init(sys.argv)
	dataSamplingOp = pyOp.ChainOperator(FFTop,tempDataSamplingOp)

	#init forcing term operator to create f
	forcingTermOp,priorTmp = wriUtilFloat.forcing_term_op_init_p(sys.argv)
	prior=priorTmp.clone()
	#mask3dOp.forward(0,priorTmp,prior)

	#init A(p)
	m_modelFloat,m_dataFloat,pressureData,tempGradioOp= Acoustic_iso_float_gradio.gradioOpInitFloat(sys.argv)
	#gradioOp = pyOp.ChainOperator(tempGradioOp,mask3dOp)
	gradioOp = tempGradioOp

	#chain operators with mask
	if(pyinfo):
		print("*** domain and range checks *** ")
		print("* KFp_w - d * ")
		print("K domain: ", dataSamplingOp.getDomain().getNdArray().shape)
		print("F range: ", FFTop.getRange().getNdArray().shape)
		print("p_w shape: ", p_modelFloat_fft.getNdArray().shape)
		print("d shape: ", p_dataFloat.getNdArray().shape)
		print("* A(m)Fp_w - f * ")
		print("A(m) domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
		print("F range: ", FFTop.getRange().getNdArray().shape)
		print("p_w shape: ", p_modelFloat_fft.getNdArray().shape)
		print("A(m) range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
		print("f shape: ", prior.getNdArray().shape)

		print("\n* A(p_t)m - f' * ")
		print("A(p_t) domain: ", gradioOp.getDomain().getNdArray().shape)
		print("m shape: ", m_modelFloat.getNdArray().shape)
		print("A(p_t) range: ", gradioOp.getRange().getNdArray().shape)
		print("f' shape: ", m_dataFloat.getNdArray().shape)

	############################# Create LCG Solvers ###############################
	# p Solver
	p_LCGsolver=LCG.LCGsolver(stop_p,logger=inv_log)
	p_prefix=prefix+"_pinv"

	# p Solver
	m_LCGsolver=LCG.LCGsolver(stop_m,logger=inv_log)
	m_prefix=prefix+"_minv"

	############################# Initial Models ###############################
	if(pyinfo): print("\n--------------------------- Reading Initial Models --------------------------------")

	# read or create initial models
	# intial wfld
	initial_p_model_file=parObject.getString("initial_p_model","None")
	if (initial_p_model_file=="None"):
		if(pyinfo): print("No initial wavefield provided... using zero wavefield")
		current_p_model=p_modelFloat_fft.cloneSpace()
		current_p_model_time=dataSamplingOp.getDomain().cloneSpace()
		current_p_model.zero(0.0)
		current_p_model_time.zero()
	else:
		initial_p_model_file_format=parObject.getString("initial_p_model_format","freq")
		if (initial_p_model_file_format=="freq"):
			if(pyinfo): print("Initial wavefield provided in freq domain: ", initial_p_model_file)
			current_p_model=genericIO.defaultIO.getVector(initial_p_model_file,storage='dataComplex')
			current_p_model_time=dataSamplingOp.getDomain().cloneSpace()
			current_p_model_time.zero()
		elif (initial_p_model_file_format=="time"):
			if(pyinfo): print("Initial wavefield provided in time domain (will convert to freq): ", initial_p_model_file)
			current_p_model_time=genericIO.defaultIO.getVector(initial_p_model_file,storage='float')
			current_p_model=SepVector.getSepVector(p_modelFloat_fft.getHyper(),storage="dataComplex")
			current_p_model.zero()
			print('here')
			FFTop.adjoint(False,current_p_model,current_p_model_time)

	initial_m_model_file=parObject.getString("initial_m_model","None")
	#initial slsq
	if (initial_m_model_file=="None"):
		if(pyinfo): print("No initial slsq provided... using water slowness")
		current_m_model=m_modelFloat.cloneSpace()
		current_m_model.set(1.0/(1500*1500))
	else:
		if(pyinfo): print("Initial slsq: ", initial_m_model_file)
		current_m_model=genericIO.defaultIO.getVector(initial_m_model_file)

	############################# Observed Data ###############################
	p_dataFile=parObject.getString("data")
	p_dataFloat=genericIO.defaultIO.getVector(p_dataFile)

	################################ DP Test ###################################
	if (parObject.getInt("dp",0)==1):
		if(pyinfo): print("\n------------------------- DP Tests ------------------------------")
		print("A(m) dot product test")
		waveEquationAcousticOp.dotTest(1)
		print("\nK dot product test")
		dataSamplingOp.dotTest(1)
		print("\nA(p) dot product test")
		if(current_p_model.norm() == 0):
			print("\nCurrent p model is zero so we will do DP test with rand p")
			rand_p_model = current_p_model.clone()
			rand_p_model.rand()
			gradioOp.get_op1().update_wfld(rand_p_model)
			gradioOp.dotTest(1)
			gradioOp.get_op1().update_wfld(current_p_model) # updates d2p/dt2
		else:
			gradioOp.get_op1().update_wfld(current_p_model) # updates d2p/dt2
			gradioOp.dotTest(1)

	############################# Regularization ###############################
	# regularization of m inversion
	minBound=parObject.getFloat("minBound", -100)
	maxBound=parObject.getFloat("maxBound", 100)
	minBoundVector = current_m_model.clone()
	maxBoundVector = current_m_model.clone()
	minBoundVector.set(minBound)
	maxBoundVector.set(maxBound)

	regOp_m_type=parObject.getString("regOp_m","None")
	if (regOp_m_type!="None"):
		# Get epsilon value from user
		epsilon_m=parObject.getFloat("eps_m_scale",1.0)*parObject.getFloat("eps_m",1.0)
		inv_log.addToLog("--- Epsilon value: %s ---"%(epsilon_m))
		maskWidth=parObject.getInt("maskWidth",0)
		mask2dOp = Mask2d.mask2d(m_modelFloat,m_modelFloat,maskWidth,m_modelFloat.getHyper().axes[0].n-maskWidth,maskWidth,m_modelFloat.getHyper().axes[1].n-maskWidth,0)

		# Spatial gradient in z-direction
		if (regOp_m_type=="zGrad"):
			if(pyinfo): print("--- Vertical gradient regularization ---")
			inv_log.addToLog("--- Vertical gradient regularization ---")
			fat=spatialDerivModule.zGradInit(sys.argv)
			tempGradOp=spatialDerivModule.zGradPython(current_m_model,current_m_model,fat)
			regOp_m = pyOp.ChainOperator(tempGradOp,mask2dOp)

		# Spatial gradient in x-direction
		if (regOp_m_type=="xGrad"):
			if(pyinfo): print("--- Horizontal gradient regularization ---")
			inv_log.addToLog("--- Horizontal gradient regularization ---")
			fat=spatialDerivModule.xGradInit(sys.argv)
			tempGradOp=spatialDerivModule.xGradPython(current_m_model,current_m_model,fat)
			regOp_m = pyOp.ChainOperator(tempGradOp,mask2dOp)

		# Sum of spatial gradients in z and x-directions
		if (regOp_m_type=="zxGrad"):
			if(pyinfo): print("--- Gradient regularization in both directions ---")
			inv_log.addToLog("--- Gradient regularization in both directions ---")
			fat=spatialDerivModule.zxGradInit(sys.argv)
			tempGradOp=spatialDerivModule.zxGradPython(current_m_model,current_m_model,fat)
			regOp_m = pyOp.ChainOperator(tempGradOp,mask2dOp)
	else:
		epsilon_m=0
		regOp_m=None
		if(pyinfo): print("--- No regularization of m inversion ---")

	############################# Gradient Editing ##############################
	gradEdit = parObject.getInt("gradEdit",0)
	if(gradEdit==1):
		gradEditOp=wriUtilFloat.grad_edit_mora
	elif(gradEdit==2):
		gradEditOp=wriUtilFloat.grad_edit_diving
	else:
		gradEditOp=None
	############################# Preconditioning ###############################
	# p inv
	precond_p=parObject.getString("precondOp_p","None")
	if(precond_p == "tpow"):
		precondStart=parObject.getFloat("precondStart",0.0)
		precondTpow=parObject.getFloat("precondTpow",1.0)
		if(pyinfo): print("--- Preconditioning p inversion w/ tpow: ",precondTpow," ---")
		precondOp_p = TpowWfld.tpow_wfld(modelFloat,modelFloat,precondTpow,precondStart)
	elif(precond_p == "gf"):
		if(pyinfo): print("--- Preconditioning p inversion w/ greens function ---")
		_,_,precondOp_p= wriUtilFloat.greens_function_op_init(sys.argv)
	else:
		if(pyinfo): print("--- No preconditioning of p inversion ---")
		inv_log.addToLog("--- No preconditioning of p inversion ---")
		precondOp_p = None

	# m inv
	precond_m=parObject.getString("precondOp_m","None")
	if(precond_m == "stack"):
		if(pyinfo): print("--- Preconditioning w/ wfld stack ---")
		inv_log.addToLog("--- Preconditioning w/ wfld stack ---")
		precondOp_m = SphericalSpreadingScale.spherical_spreading_scale_wfld(modelInit,modelInit,pressureData)
	else:
		if(pyinfo): print("--- No preconditioning of m inversion---")
		inv_log.addToLog("--- No preconditioning of m inversion ---")
		precondOp_m = None

	############################# Begin Inversion ###############################
	if(pyinfo): print("\n--------------------------- Checking restart --------------------------------")

	# check restart
	restartIter=parObject.getInt("restartIter",-1)
	if(restartIter==-1):
		print("no restart")
		pFinished=parObject.getInt("pFinished",0)
		restartIter=0
	else:
		pFinished=parObject.getInt("pFinished",0)
		if(pFinished==1):
			current_p_model=genericIO.defaultIO.getVector(p_prefix+"_iter"+str(restartIter)+"_inv_mod.H")
		else:
			current_p_model=genericIO.defaultIO.getVector(p_prefix+"_iter"+str(restartIter-1)+"_inv_mod.H")
		if(restartIter!=0):
			current_m_model=genericIO.defaultIO.getVector(m_prefix+"_iter"+str(restartIter-1)+"_inv_mod.H")
		print("restarting at iteration ",restartIter,". Wavefield already reconstructed=",pFinished)

	############################# Evaluate epsilon ###############################
	#need to set earth model in wave equation operator
	waveEquationAcousticOp.op2.update_slsq(current_m_model)
	# Evaluate Epsilon for p inversion
	if (epsilonEval==1):
		if(pyinfo): print("--- Epsilon evaluation ---")
		inv_log.addToLog("--- Epsilon evaluation ---")
		epsilon_p = wriUtilFloat.evaluate_epsilon(current_p_model,p_dataFloat,prior,dataSamplingOp,waveEquationAcousticOp,parObject)
	else:
		epsilon_p=parObject.getFloat("eps_p_scale",1.0)*parObject.getFloat("eps_p",1.0)
	if(pyinfo): print("--- Epsilon value: ",epsilon_p," ---")
	inv_log.addToLog("--- Epsilon value: %s ---"%(epsilon_p))

	# for number of outer loops
	for iteration in range(restartIter,nIter):
		if(pyinfo): print("\n---------------------- Running Iteration ",iteration," ---------------------------")
		inv_log.addToLog("\n---------------------- Running Iteration "+str(iteration)+" ---------------------------")

		if(pFinished==0):
			# minimize ||Kp-d||+e^2/2||A(m)p-f|| w.r.t. p
			# update m
			waveEquationAcousticOp.op2.update_slsq(current_m_model)
			#re-evaluate epsilon
			if(parObject.getInt("reEvalEpsilon",0)!=0):
				epsilon_p = wriUtilFloat.evaluate_epsilon(current_p_model,p_dataFloat,prior,dataSamplingOp,waveEquationAcousticOp,parObject)
				print("--------------- Re-evalute epsilon to: ",epsilon_p," -------------")
				inv_log.addToLog("\n---------------   Re-evalute epsilon to: "+str(epsilon_p)+" ----------------------")
			elif(parObject.getFloat("reScaleEpsilon",0)!=0 and iteration!=0 ):
				scale = parObject.getFloat("reScaleEpsilon",0)
				epsilon_p = epsilon_p * scale
				print("--------------- Re-scale epsilon by ",scale,". Epsiilon is now: ",epsilon_p," -------------")
				inv_log.addToLog("--------------- Re-scale epsilon by "+str(scale)+". Epsiilon is now: "+str(epsilon_p)+" -------------")
			# reinit L2 problem with new initial p
			p_invProb=Prblm.ProblemL2LinearReg(current_p_model,p_dataFloat,dataSamplingOp,epsilon_p,reg_op=waveEquationAcousticOp,prior_model=prior,prec=precondOp_p)
			# update prefix and solver defaults
			p_prefix_cur=p_prefix+"_iter"+str(iteration)
			p_LCGsolver.setDefaults(save_obj=saveObj_p,save_res=saveRes_p,save_grad=saveGrad_p,save_model=saveModel_p,prefix=p_prefix_cur,iter_buffer_size=bufferSize_p,iter_sampling=iterSampling_p,flush_memory=flushMemory)
			# run LCG solver
			p_LCGsolver.run(p_invProb,verbose=True)
			# update current p model
			current_p_model=p_LCGsolver.inv_model
			FFTop.forward(False,current_p_model,current_p_model_time)
		pFinished=0
		# minimize ||A(p)m-f|| w.r.t. m
		# update p and f
		#m_dataFloat,_ = Acoustic_iso_float_gradio.update_data(current_p_model,sys.argv) #f is actaully f+Lapl(p) so we need to update
		#gradioOp.get_op1().update_wfld(current_p_model) # updates d2p/dt2
		# create new gradioOp
		_,m_dataFloat,pressureData,gradioOp= Acoustic_iso_float_gradio.gradioOpInitFloat_givenPressure(current_p_model_time,sys.argv)
		#update prec
		# reinit L2 problem with new initial m
		m_invProb=Prblm.ProblemL2LinearReg(current_m_model,m_dataFloat,gradioOp,epsilon_m,minBound=minBoundVector,maxBound=maxBoundVector,reg_op=regOp_m,prec=precondOp_m)
		# update prefix and solver defaults
		m_prefix_cur=m_prefix+"_iter"+str(iteration)
		m_LCGsolver.setDefaults(save_obj=saveObj_m,save_res=saveRes_m,save_grad=saveGrad_m,save_model=saveModel_m,prefix=m_prefix_cur,iter_buffer_size=bufferSize_m,iter_sampling=iterSampling_m,flush_memory=flushMemory)
		# run LCG solver
		m_LCGsolver.run(m_invProb,verbose=True)
		#update current m model
		current_m_model=m_LCGsolver.inv_model

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("--------------------------- All done ------------------------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")

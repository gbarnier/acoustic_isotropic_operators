#!/usr/bin/env python3.6
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
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
import inversionUtils
import wriUtilFloat
import TpowWfld
import Mask3d 
import Mask2d 
import Acoustic_iso_float_gradio
import spatialDerivModule
import SphericalSpreadingScale
from sys_util import logger

if __name__ == '__main__':

	# io stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	pyinfo=parObject.getInt("pyinfo",1)
	epsilonEval=parObject.getInt("epsilonEval",0)
	regType=parObject.getString("reg","None")
	reg=0
	if (regType != "None"): reg=1
	# Initialize parameters for inversion
	stop,logFile,saveObj,saveRes,saveGrad,saveModel,prefix,bufferSize,iterSampling,restartFolder,flushMemory,info=inversionUtils.inversionInit(sys.argv)
	# Logger
	inv_log = logger(logFile)

	if(pyinfo): print("-------------------------------------------------------------------")
	if(pyinfo): print("------------------ gradiometry --------------")
	if(pyinfo): print("-------------------------------------------------------------------\n")
	inv_log.addToLog("------------------ gradiometry --------------")

	############################# Initialization ###############################
	# Wave equation op init
	if(pyinfo): print("--------------------------- Gradio op init --------------------------------")
	modelFloat,dataFloat,pressureData,tempGradioOp= Acoustic_iso_float_gradio.gradioOpInitFloat(sys.argv)
	maskWidth=parObject.getInt("maskWidth",0)
	mask3dOp = Mask3d.mask3d(dataFloat,dataFloat,maskWidth,dataFloat.getHyper().axes[0].n-maskWidth,maskWidth,dataFloat.getHyper().axes[1].n-maskWidth,0,dataFloat.getHyper().axes[2].n,0)
	mask2dOp = Mask2d.mask2d(modelFloat,modelFloat,maskWidth,modelFloat.getHyper().axes[0].n-maskWidth,maskWidth,dataFloat.getHyper().axes[1].n-maskWidth,0)
	gradioOp = pyOp.ChainOperator(tempGradioOp,mask3dOp)

	################################ DP Test ###################################
	if (parObject.getInt("dp",0)==1):
		print("Gradiometry DP test:")
		print("\nModel op dp test:")
		gradioOp.dotTest(1)

	############################# Read files ###################################
	# Read initial model
	modelInitFile=parObject.getString("modelInit","None")
	if (modelInitFile=="None"):
		modelInit=modelFloat.clone()
		modelInit.set(1.0/(1500*1500))
	else:
		print("Using intitial model: ",modelInitFile)
		modelInit=genericIO.defaultIO.getVector(modelInitFile)

	print("*** domain and range checks *** ")
	print("* A(p)m - f' * ")
	print("A(p) domain: ", gradioOp.getDomain().getNdArray().shape)
	print("m shape: ", modelInit.getNdArray().shape)
	print("A(p) range: ", gradioOp.getRange().getNdArray().shape)
	print("f' shape: ", dataFloat.getNdArray().shape)

	############################# Regularization ###############################
	# Read min/max bound 
	minBound=parObject.getFloat("minBound", -1)
	maxBound=parObject.getFloat("maxBound", -1)
	minBoundVector = modelInit.clone()
	maxBoundVector = modelInit.clone()
	if (minBound != -1):
		minBoundVector.set(minBound)
	else:
		minBoundVector.set(-100.0) #default min is 0 m/s

	if (maxBound != -1):
		maxBoundVector.set(maxBound)
	else:
		maxBoundVector.set(100.0) #default max is 6000 m/s

	if (reg==1):
		# Get epsilon value from user
		epsilon=parObject.getFloat("epsScale",1.0)*parObject.getFloat("eps",1.0)
		inv_log.addToLog("--- Epsilon value: %s ---"%(epsilon))

		# Spatial gradient in z-direction
		if (regType=="zGrad"):
			if(pyinfo): print("--- Vertical gradient regularization ---")
			inv_log.addToLog("--- Vertical gradient regularization ---")
			fat=spatialDerivModule.zGradInit(sys.argv)
			tempGradOp=spatialDerivModule.zGradPython(modelInit,modelInit,fat)
			regOp = pyOp.ChainOperator(tempGradOp,mask2dOp)

		# Spatial gradient in x-direction
		if (regType=="xGrad"):
			if(pyinfo): print("--- Horizontal gradient regularization ---")
			inv_log.addToLog("--- Horizontal gradient regularization ---")
			fat=spatialDerivModule.xGradInit(sys.argv)
			tempGradOp=spatialDerivModule.xGradPython(modelInit,modelInit,fat)
			regOp = pyOp.ChainOperator(tempGradOp,mask2dOp)

		# Sum of spatial gradients in z and x-directions
		if (regType=="zxGrad"):
			if(pyinfo): print("--- Gradient regularization in both directions ---")
			inv_log.addToLog("--- Gradient regularization in both directions ---")
			fat=spatialDerivModule.zxGradInit(sys.argv)
			tempGradOp=spatialDerivModule.zxGradPython(modelInit,modelInit,fat)
			regOp = pyOp.ChainOperator(tempGradOp,mask2dOp)

		#init with regOp
		#check for preconditioning
		tpow=parObject.getFloat("tpowPrecond",0.0)
		if(tpow != 0.0):
			if(pyinfo): print("--- Preconditioning w/ wfld stack ---") 
			inv_log.addToLog("--- Preconditioning w/ wfld stack ---")
			sphericalTpow= SphericalSpreadingScale.spherical_spreading_scale_wfld(modelInit,modelInit,pressureData)
			invProb=Prblm.ProblemL2LinearReg(modelInit,dataFloat,gradioOp,epsilon,minBound=minBoundVector,maxBound=maxBoundVector,reg_op=regOp,prec=sphericalTpow)
		else:
			if(pyinfo): print("--- No preconditioning ---")
			inv_log.addToLog("--- No preconditioning ---")
			invProb=Prblm.ProblemL2LinearReg(modelInit,dataFloat,gradioOp,epsilon,minBound=minBoundVector,maxBound=maxBoundVector,reg_op=regOp)
	

		# Evaluate Epsilon
		if (epsilonEval==1):
			if(pyinfo): print("--- Epsilon evaluation ---")
			inv_log.addToLog("--- Epsilon evaluation ---")
			epsilonOut=invProb.estimate_epsilon()
			if(pyinfo): print("--- Epsilon value: ",epsilonOut," ---")
			inv_log.addToLog("--- Epsilon value: %s ---"%(epsilonOut))
			quit()

	# No regularization
	else:
		if(pyinfo): print("--- No regularization ---")
		inv_log.addToLog("--- No regularization ---")
		#check for preconditioning
		tpow=parObject.getFloat("tpowPrecond",0.0)
		if(tpow != 0.0):
			if(pyinfo): print("--- Preconditioning w/ wfld stack ---") 
			inv_log.addToLog("--- Preconditioning w/ wfld stack ---")
			sphericalTpow= SphericalSpreadingScale.spherical_spreading_scale_wfld(modelInit,modelInit,pressureData)
			invProb=Prblm.ProblemL2Linear(modelInit,dataFloat,gradioOp,minBoundVector,maxBoundVector,prec=sphericalTpow)
		else:
			if(pyinfo): print("--- No preconditioning ---")
			inv_log.addToLog("--- No preconditioning ---")
			invProb=Prblm.ProblemL2Linear(modelInit,dataFloat,gradioOp,minBoundVector,maxBoundVector)

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

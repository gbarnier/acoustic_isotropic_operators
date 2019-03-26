#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Solver library
# import pyOperator as pyOp
# import pyNLCGsolver as NLCG
# import pyProblem as Prblm
import pyStopperBase as Stopper
# from sys_util import logger

def inversionInit(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	par=ioDef.getParamObj()

	# Stopper
	nIter=par.getInt("nIter")
	stop=Stopper.BasicStopper(niter=par.getInt("nIter"))

	# Inversion Folder
	folder=par.getString("folder")
	if (os.path.isdir(folder)==False): os.mkdir(folder)
	prefix=par.getString("prefix","None")
	if (prefix=="None"): prefix=folder
	invPrefix=folder+"/"+prefix
	logFile=invPrefix+"_logFile"

	# Recording parameters
	bufferSize=par.getInt("bufferSize",3)
	if (bufferSize==0): bufferSize=None
	iterSampling=par.getInt("iterSampling",10)
	restartFolder=par.getString("restartFolder","None")
	flushMemory=par.getInt("flushMemory",0)

	# Inversion components to save
	saveObj=par.getInt("saveObj",1)
	saveRes=par.getInt("saveRes",1)
	saveGrad=par.getInt("saveGrad",1)
	saveModel=par.getInt("saveModel",1)

	# Info
	info=par.getInt("info",1)

	return stop,logFile,saveObj,saveRes,saveGrad,saveModel,invPrefix,bufferSize,iterSampling,restartFolder,flushMemory,info

def inversionVpInit(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	par=ioDef.getParamObj()

	################################# Nonlinear ################################
	# Usual shits
	nIterNl=par.getInt("nIterNl")
	stopNl=Stopper.BasicStopper(niter=nIterNl)
	folderNl=par.getString("folderNl")
	if (os.path.isdir(folderNl)==False): os.mkdir(folderNl)
	prefixNl=par.getString("prefixNl","None")
	if (prefixNl=="None"): prefixNl=folderNl
	invPrefixNl=folderNl+"/"+prefixNl
	logFileNl=invPrefixNl+"_logFile"

	# Epsilon
	epsilon=par.getFloat("epsilon")

	# Recording parameters
	bufferSizeNl=par.getInt("bufferSizeNl",1)
	if (bufferSizeNl==0): bufferSizeNl=None
	iterSamplingNl=par.getInt("iterSamplingNl",1)
	restartFolderNl=par.getString("restartFolderNl","None")
	flushMemoryNl=par.getInt("flushMemoryNl",0)

	# Inversion components to save
	saveObjNl=par.getInt("saveObjNl",1)
	saveResNl=par.getInt("saveResNl",1)
	saveGradNl=par.getInt("saveGradNl",1)
	saveModelNl=par.getInt("saveModelNl",1)

	################################# Linear ###################################
	# Usual shits
	nIterLin=par.getInt("nIterLin")
	stopLin=Stopper.BasicStopper(niter=nIterLin)
	folderLin=folderNl+"/varProFolder"
	if (os.path.isdir(folderLin)==False): os.mkdir(folderLin)
	invPrefixLin=folderLin+"/varPro"
	logFileLin=invPrefixLin+"_logFile"

	# Recording parameters
	bufferSizeLin=par.getInt("bufferSizeLin",1)
	if (bufferSizeLin==0): bufferSizeLin=None
	iterSamplingLin=par.getInt("iterSamplingLin",1)
	restartFolderLin=par.getString("restartFolderLin","None")
	flushMemoryLin=par.getInt("flushMemoryLin",0)

	# Inversion components to save
	saveObjLin=par.getInt("saveObjLin",1)
	saveResLin=par.getInt("saveResLin",1)
	saveGradLin=par.getInt("saveGradLin",1)
	saveModelLin=par.getInt("saveModelLin",1)

	# Info
	info=par.getInt("info",1)

	return stopNl,logFileNl,saveObjNl,saveResNl,saveGradNl,saveModelNl,invPrefixNl,bufferSizeNl,iterSamplingNl,restartFolderNl,flushMemoryNl,stopLin,logFileLin,saveObjLin,saveResLin,saveGradLin,saveModelLin,invPrefixLin,bufferSizeLin,iterSamplingLin,restartFolderLin,flushMemoryLin,epsilon,info

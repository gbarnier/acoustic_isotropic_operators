#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os

# Solver library
import pyOperator as pyOp
import pyNLCGsolver as NLCG
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger

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

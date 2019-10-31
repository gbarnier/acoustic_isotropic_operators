#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os
import pyStopperBase as Stopper

def inversionFullWRIInit(args):

	# IO object
	par=genericIO.io(params=args)

	# Stopper
	nIter=par.getInt("nIter")
	nIter_p=par.getInt("nIter_p")
	stop_p=Stopper.BasicStopper(nIter_p)
	nIter_m=par.getInt("nIter_m")
	stop_m=Stopper.BasicStopper(nIter_m)

	# Inversion Folder
	folder=par.getString("folder")
	if (os.path.isdir(folder)==False): os.mkdir(folder)
	prefix=par.getString("prefix","None")
	if (prefix=="None"): prefix=folder
	invPrefix=folder+"/"+prefix
	logFile=invPrefix+"_logFile"

	# Recording parameters
	bufferSize_p=par.getInt("bufferSize_p",3)
	if (bufferSize_p==0): bufferSize_p=None
	iterSampling_p=par.getInt("iterSampling_p",1000)
	bufferSize_m=par.getInt("bufferSize_m",3)
	if (bufferSize_m==0): bufferSize_m=None
	iterSampling_m=par.getInt("iterSampling_m",20)
	restartFolder=par.getString("restartFolder","None")
	flushMemory=par.getInt("flushMemory",0)

	# Inversion components to save
	saveObj_p=par.getInt("saveObj_p",1)
	saveRes_p=par.getInt("saveRes_p",1)
	saveGrad_p=par.getInt("saveGrad_p",1)
	saveModel_p=par.getInt("saveModel_p",1)
	saveObj_m=par.getInt("saveObj_m",1)
	saveRes_m=par.getInt("saveRes_m",1)
	saveGrad_m=par.getInt("saveGrad_m",1)
	saveModel_m=par.getInt("saveModel_m",1)

	# Info
	info=par.getInt("info",1)

	return nIter,stop_m,stop_p,logFile,saveObj_p,saveRes_p,saveGrad_p,saveModel_p,saveObj_m,saveRes_m,saveGrad_m,saveModel_m,invPrefix,bufferSize_p,iterSampling_p,bufferSize_m,iterSampling_m,restartFolder,flushMemory,info

def inversionInit(args):

	# IO object
	par=genericIO.io(params=args)

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

	# IO object
	par=genericIO.io(params=args)

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
	invPrefixLin=folderLin+"/varPro_"+prefixNl
	logFileLin=invPrefixLin+"_logFile"

	# Recording parameters
	bufferSizeLin=par.getInt("bufferSizeLin",3)
	if (bufferSizeLin==0): bufferSizeLin=None
	iterSamplingLin=par.getInt("iterSamplingLin",10)
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

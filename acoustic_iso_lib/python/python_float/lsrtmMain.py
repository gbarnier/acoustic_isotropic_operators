#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import numpy as np
import time
import sys

# Solver library
import pyOperator as pyOp
import pyLCGsolver as LCG
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger

# Template for LSRTM workflow
if __name__ == '__main__':

    print("-------------------------------------------------------------------")
    print("---------------------- Running least-squares migration ------------")
    print("-------------------------------------------------------------------\n")

    ############################# Initialization ###############################
    # Born
    modelStartFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_float.BornOpInitFloat(sys.argv)

    ############################# Read files ###################################
    # Starting model
    modelStartFloatFile=parObject.getString("modelStart")
    modelStartFloat=genericIO.defaultIO.getVector(modelStartFloatFile,ndims=2)

    # Data
    dataFile=parObject.getString("data")
    dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)

    ############################# Instanciation ################################
    # Born
    BornOp=Acoustic_iso_float.BornShotsGpu(modelStartFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

    ############################# Solver #######################################
    # Problem
    lsrtmProb=Prblm.ProblemL2Linear(modelStartFloat,dataFloat,BornOp)

    # Stopper
    stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

    # Solver
    logFile=parObject.getString("log","lsrtmLog")
    invPrefix=parObject.getString("prefix","lsrtmPrefix")
    LCGsolver=LCG.LCGsolver(stop,logger=logger(logFile))
    LCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

    # Run solver
    LCGsolver.run(lsrtmProb,verbose=True)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import numpy as np
import time
import sys

# Solver library
import import pyOperator as pyOp
import pyNLCGsolver as NLCG
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logge

# Plotting library
import matplotlib.pyplot as plt
import sepPlot

# Template for extended LSRTM workflow
if __name__ == '__main__':

    print("-------------------------------------------------------------------")
    print("--------------- Running extended least-squares migration ----------")
    print("-------------------------------------------------------------------\n")

    ############################# Initialization ###############################
    modelStartFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_double.BornExtOpInitFloat(sys.argv)

    ############################# Read files ###################################
    # Starting model
    modelStartFloatFile=parObject.getString("modelStart")
    modelStartFloat=genericIO.defaultIO.getVector(modelStartFloatFile,ndims=3)

    # Data
    dataFile=parObject.getString("data")
    dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)

    ############################# Instanciation ################################
    # Born extended
    BornExtOp=Acoustic_iso_float.BornExtShotsGpu(modelStartFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

    ############################# Solver #######################################
    # Problem
    extLsrtmProb=Prblm.ProblemL2Linear(modelStartFloat,dataFloat,BornExtOp)

    # Stopper
    stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

    # Solver
    logFile=parObject.getString("log")
    LCGsolver=LCG.LCGsolver(Stop,logger=logger(logFile))
    LCGsolver.setDefaults(iter_buffer=None,iter_sampling=1)

    # Run solver
    LCGsolver.run(extLsrtmProb)

    ############################# Results ######################################
    # Inverted model
    extLsrtmModelFloat=extLsrtmProb.model

    # Write model to disk
    extLsrtmModelFile=parObject.getString("invertedModel")
    genericIO.defaultIO.writeVector(extLsrtmModelFile,extLsrtmModelFloat)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_double
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

# Template for LSRTM workflow
if __name__ == '__main__':

    print("-------------------------------------------------------------------")
    print("--------------- Running extended least-squares migration ----------")
    print("-------------------------------------------------------------------\n")

    ############################# Initialization ###############################
    modelStartDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_double.BornExtOpInitDouble(sys.argv)

    ############################# Read files ###################################
    # Starting model
    modelStartFloatFile=parObject.getString("modelStart")
    modelStartFloat=genericIO.defaultIO.getVector(modelStartFloatFile,ndims=3)
    modelStartFloatNd=modelStartFloat.getNdArray()
    modelStartDoubleNd=modelStartDouble.getNdArray()
    modelStartDoubleNd[:]=modelStartFloatNd

    # Data
    dataFile=parObject.getString("data")
    dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)
    dataFloatNd=dataFloat.getNdArray()
    dataDoubleNd=dataDouble.getNdArray()
    dataDoubleNd[:]=dataFloatNd

    ############################# Instanciation ################################
    # Born extended
    BornExtOp=Acoustic_iso_double.BornExtShotsGpu(modelStartDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

    ############################# Solver #######################################
    # Problem
    extLsrtmProb=Prblm.ProblemL2Linear(modelStartDouble,dataDouble,BornExtOp)

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
    extLsrtmModelDouble=extLsrtmProb.model.clone()

    # Write model to disk
    extLsrtmModelFloat=SepVector.getSepVector(extLsrtmModelDouble.getHyper(),storage="dataFloat")
    extLsrtmModelFloatNp=extLsrtmModelFloat.getNdArray()
    extLsrtmModelDoubleNp=extLsrtmModelDouble.getNdArray()
    lsrtmModelFloatNp[:]=extLsrtmModelDoubleNp
    extLsrtmModelFile=parObject.getString("invertedModel")
    genericIO.defaultIO.writeVector(extLsrtmModelFile,extLsrtmModelFloat)

    # Write objective function


    # Write data residuals

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

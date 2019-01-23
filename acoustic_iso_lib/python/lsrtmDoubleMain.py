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
    print("---------------------- Running least-squares migration ------------")
    print("-------------------------------------------------------------------\n")

    ############################# Initialization ###############################
    modelStartDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_double.BornOpInitDouble(sys.argv)

    ############################# Read files ###################################
    # Starting model
    modelStartFloatFile=parObject.getString("modelStart")
    modelStartFloat=genericIO.defaultIO.getVector(modelStartFloatFile,ndims=2)
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
    # Born
    BornOp=Acoustic_iso_double.BornShotsGpu(modelStartDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

    ############################# Solver #######################################
    # Problem
    lsrtmProb=Prblm.ProblemL2Linear(modelStartDouble,dataDouble,BornOp)

    # Stopper
    stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

    # Solver
    logFile=parObject.getString("log")
    LCGsolver=LCG.LCGsolver(Stop,logger=logger(logFile))
    LCGsolver.setDefaults(iter_buffer=None,iter_sampling=1)

    # Run solver
    LCGsolver.run(lsrtmProb)

    ############################# Results ######################################
    # Inverted model
    lsrtmModelDouble=lsrtmProb.model.clone()

    # Write model to disk
    lsrtmModelFloat=SepVector.getSepVector(lsrtmModelDouble.getHyper(),storage="dataFloat")
    lsrtmModelFloatNp=lsrtmModelFloat.getNdArray()
    lsrtmModelDoubleNp=lsrtmModelDouble.getNdArray()
    lsrtmModelFloatNp[:]=lsrtmModelDoubleNp
    lsrtmModelFile=parObject.getString("invertedModel")
    genericIO.defaultIO.writeVector(lsrtmModelFile,lsrtmModelFloat)

    # Write objective function


    # Write data residuals

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

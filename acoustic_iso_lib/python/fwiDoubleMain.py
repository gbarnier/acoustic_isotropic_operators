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

# Template for FWI workflow
if __name__ == '__main__':

    print("-------------------------------------------------------------------")
    print("--------------------------- Running FWI ---------------------------")
    print("-------------------------------------------------------------------\n")

    ############################# Initialization ###############################
    # Nonlinear
    waveletDouble,dataDouble,modelStartFwiDouble,parObject,sourcesVector,receiversVector=Acoustic_iso_double.nonlinearOpInitDouble(sys.argv)

    # Born
    _,_,_,_,_,sourcesSignalsVector,_=Acoustic_iso_double.BornOpInitDouble(sys.argv)

    # Other operators

    ############################# Read files ###################################
    # Seismic source
    waveletFile=parObject.getString("sources")
    waveletFloat=genericIO.defaultIO.getVector(waveletFile,ndims=3)
    waveletFloatNd=waveletFloat.getNdArray()
    waveletDoubleNd=waveletDouble.getNdArray()
    waveletDoubleNd[:]=waveletFloatNd

    # Data
    dataFile=parObject.getString("data")
    dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)
    dataFloatNd=dataFloat.getNdArray()
    dataDoubleNd=dataDouble.getNdArray()
    dataDoubleNd[:]=dataFloatNd

    ############################# Instanciation ################################
    # Nonlinear
    nonlinearVelocityOp=Acoustic_iso_double.nonlinearVelocityPropShotsGpu(modelStartFwiDouble,dataDouble,waveletDouble,parObject,sourcesVector,receiversVector)

    # Born
    BornOp=Acoustic_iso_double.BornShotsGpu(modelStartFwiDouble,dataDouble,modelStartFwiDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

    # FWI
    fwiOp=pyOp.NonLinearOperator(nonlinearVelocityOp,BornOp,BornOp.setVel)

    # Other operators

    ############################# Solver #######################################
    # L2-norm nonlinear problem
    invertedModel=modelStartFwiDouble.clone()
    fwiProb=Prblm.ProblemL2NonLinear(invertedModel,dataDouble,fwiOp)

    # Stopper
    stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

    # Solver
    logFile=parObject.getString("log")
    NLCGsolver=NLCG.NLCGsolver(stop,logger=logger(logFile))
    NLCGsolver.setDefaults(iter_buffer=None)

    # Run solver
    NLCGsolver.run(fwiProb)

    ############################# Results ######################################
    # Inverted model
    fwiModelDouble=fwiProb.model.clone()

    # Write model to disk
    fwiModelFloat=SepVector.getSepVector(fwiModelDouble.getHyper(),storage="dataFloat")
    fwiModelFloatNp=fwiModelFloat.getNdArray()
    fwiModelDoubleNp=fwiModelDouble.getNdArray()
    fwiModelFloatNp[:]=fwiModelDouble
    fwiModelFile=parObject.getString("invertedModel")
    genericIO.defaultIO.writeVector(fwiModelFile,fwiModelFloat)

    # Write objective functions to disk


    # Write data residuals to disk

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

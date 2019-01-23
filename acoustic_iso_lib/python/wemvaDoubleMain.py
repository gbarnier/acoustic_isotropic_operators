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
    print("--------------------------- Running WEMVA -------------------------")
    print("-------------------------------------------------------------------\n")

    ############################# Initialization ###############################
    # Born extended adjoint
    modelStartDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector=Acoustic_iso_double.BornExtOpInitDouble(sys.argv)

    # WEMVA
    modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector,wemvaData=Acoustic_iso_double.wemvaExtOpInitDouble(sys.argv)

    # Other operators
    # ****** Initialize DSO ****** #

    ############################# Read files ###################################
    # Starting model
    modelStartFloatFile=parObject.getString("modelStart")
    modelStartFloat=genericIO.defaultIO.getVector(modelStartFloatFile,ndims=2)
    modelStartFloatNd=modelStartFloat.getNdArray()
    modelStartDoubleNd=modelStartDouble.getNdArray()
    modelStartDoubleNd[:]=modelStartFloatNd

    ############################# Instanciation ################################
    # Born extended
    BornExtOp=Acoustic_iso_double.BornExtShotsGpu(modelStartDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

    # Wemva
    wemvaExtOp=Acoustic_iso_double.wemvaExtPropShotsGpu(modelStartDouble,dataDouble,waveletDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector,wemvaData)

    # Wemva problem
    wemvaOp=pyOp.NonLinearOperator(nonlinearVelocityOp,BornOp,BornOp.setVel)

    # Other operators

    ############################# Solver #######################################
    # L2-norm nonlinear problem
    invertedModel=modelStartDouble.clone()
    wemvaProb=Prblm.ProblemL2NonLinear(invertedModel,dataDouble,wemvaOp)

    # Stopper
    stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

    # Solver
    logFile=parObject.getString("log")
    NLCGsolver=NLCG.NLCGsolver(stop,logger=logger(logFile))
    NLCGsolver.setDefaults(iter_buffer=None)

    # Run solver
    NLCGsolver.run(wemvaProb)

    ############################# Results ######################################
    # Inverted model
    wemvaModelDouble=wemvaProb.model.clone()

    # Write model to disk
    wemvaModelFloat=SepVector.getSepVector(wemvaModelDouble.getHyper(),storage="dataFloat")
    wemvaModelFloatNp=wemvaModelFloat.getNdArray()
    wemvaModelDoubleNp=wemvaModelDouble.getNdArray()
    wemvaModelFloatNp[:]=wemvaModelDouble
    wemvaModelFile=parObject.getString("invertedModel")
    genericIO.defaultIO.writeVector(wemvaModelFile,wemvaModelFloat)

    # Write objective functions to disk


    # Write data residuals to disk

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

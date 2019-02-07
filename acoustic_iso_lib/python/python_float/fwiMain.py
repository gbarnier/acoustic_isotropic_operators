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
import pyNLCGsolver as NLCG
import pyProblem as Prblm
import pyStopperBase as Stopper
from sys_util import logger

# Template for FWI workflow
if __name__ == '__main__':

    print("-------------------------------------------------------------------")
    print("--------------------------- Running FWI ---------------------------")
    print("-------------------------------------------------------------------\n")

    ############################# Initialization ###############################
    # Nonlinear
    waveletFloat,dataFloat,modelStartFwiFloat,parObject,sourcesVector,receiversVector=Acoustic_iso_float.nonlinearOpInitFloat(sys.argv)

    # Born
    _,_,_,_,_,sourcesSignalsVector,_=Acoustic_iso_float.BornOpInitFloat(sys.argv)

    ############################# Read files ###################################
    # Seismic source
    waveletFile=parObject.getString("sources")
    waveletFloat=genericIO.defaultIO.getVector(waveletFile,ndims=3)

    # Data
    dataFile=parObject.getString("data")
    dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)

    ############################# Instanciation ################################
    # Nonlinear
    nonlinearVelocityOp=Acoustic_iso_float.nonlinearVelocityPropShotsGpu(modelStartFwiFloat,dataFloat,waveletFloat,parObject,sourcesVector,receiversVector)

    # Born
    BornOp=Acoustic_iso_float.BornShotsGpu(modelStartFwiFloat,dataFloat,modelStartFwiFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

    # FWI
    fwiOp=pyOp.NonLinearOperator(nonlinearVelocityOp,BornOp,BornOp.setVel)

    # Case where we apply an additional operator to the residuals
    # addOperator=parObject.getString("addOperator")
    # if (addOperator == 1):
    #
    #     # Additional operator
    #     dataTaperOp=dataTaperFloatModule.dataTaper(dataFloat,dataFloat,parObject.getFloat("maxOffset",0),parObject.getFloat("exp",2),parObject.getFloat("taperWidth",0),dataDouble.getHyper(),parObject.getString("muteType","offset"))
    #
    #     # Nonlinear operator chain
    #     chainNonlinear=pyOpterator.ChainOperator(nonlinearVelocityOp,dataTaperOp)
    #
    #     # Linear operator chain
    #     chainLinear=pyOpterator.ChainOperator(BornOp,dataTaperOp)
    #
    #     # FWI
    #     fwiOp=pyOp.NonLinearOperator(chainNonlinear,chainLinear,BornOp.setVel)
    #
    # else:
    #
    #     # FWI
    #     fwiOp=pyOp.NonLinearOperator(nonlinearVelocityOp,BornOp,BornOp.setVel)


    ############################# Solver #######################################
    # L2-norm nonlinear problem
    invertedModel=modelStartFwiFloat.clone()
    fwiProb=Prblm.ProblemL2NonLinear(invertedModel,dataFloat,fwiOp)

    # Stopper
    stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

    # Solver
    logFile=parObject.getString("log","logDefault")
    invPrefix=parObject.getString("prefix","fwiPrefix")
    NLCGsolver=NLCG.NLCGsolver(stop,logger=logger(logFile))
    NLCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

    # Run solver
    NLCGsolver.run(fwiProb,verbose=True)

    ############################# Results ######################################
    # Inverted model
    # fwiModelFloat=NLCGsolver.model[9] # Inverted model
    #
    # # Gradient
    # # fwiGradFloat=SepVector.getSepVector(lsrtmModelDouble.getHyper(),storage="dataFloat")
    # fwiGradFloat=NLCGsolver.grad[9] # Gradient
    #
    # # Write model to disk
    # fwiModelFile=parObject.getString("invertedModel")
    # genericIO.defaultIO.writeVector(fwiModelFile,fwiModelFloat)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

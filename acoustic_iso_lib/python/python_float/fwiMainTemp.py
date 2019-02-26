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

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	par=ioDef.getParamObj()
    dataTaper=par.getString("dataTaper",0)
    spline=par.getString("spline",0)

    ############################# Initialization ###############################
    # Nonlinear
    wavelet,data,modelStartFwi,parObject,sourcesVector,receiversVector=Acoustic_iso_float.nonlinearOpInitFloat(sys.argv)

    # Born
    _,_,_,_,_,sourcesSignalsVector,_=Acoustic_iso_float.BornOpInitFloat(sys.argv)

    # Data tapering
    if (dataTaper==1):
        t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset=dataTaperModule.dataTaperInit(sys.argv)

	# Splines
    if (spline==1):
    	modelStartSpline,_,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat=interpBSpline2dModule.bSpline2dInit(sys.argv)

    ############################# Read files ###################################
    # Seismic source
    waveletFile=parObject.getString("sources")
    waveletFloat=genericIO.defaultIO.getVector(waveletFile,ndims=3)

    # Data
    dataFile=parObject.getString("data")
    data=genericIO.defaultIO.getVector(dataFile,ndims=3)

    ############################# Instanciation ################################
    # Nonlinear
    nonlinearVelocityOp=Acoustic_iso_float.nonlinearVelocityPropShotsGpu(modelStartFwiFloat,data,waveletFloat,parObject,sourcesVector,receiversVector)

    # Born
    BornOp=Acoustic_iso_float.BornShotsGpu(modelStartFwiFloat,dataFloat,modelStartFwiFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector)

    # Data tapering only
    if (dataTaper==1 and spline==0):

        print("-------------------------------------------------------------------")
        print("--------------------------- Running FWI ---------------------------")
        print("---------------------------- Data taper ---------------------------")
        print("-------------------------------------------------------------------\n")

        # Instanciate tapering operator
        dataTaperOp=dataTaperModule.datTaper(model,model,t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,model.getHyper(),time,offset)

        # Apply a forward to the data
        dataTaperOb.forward(False,data,dataTapered)
        data=dataTapered

        # Nonlinear operator chain
        fwiTaperNonlinearOp=pyOpterator.ChainOperator(nonlinearVelocityOp,dataTaperOb)

        # Linear operator chain
        fwiTaperLinearOp=pyOpterator.ChainOperator(BornOp,dataTaperOp)

        # FWI
        fwiOp=pyOp.NonLinearOperator(fwiTaperNonlinearOp,fwiTaperLinearOp,BornOp.setVel)

    # Spline interpolation only
    if (dataTaper==0 and spline==1):

        print("-------------------------------------------------------------------")
        print("--------------------------- Running FWI ---------------------------")
        print("---------------------- Spline interpolation -----------------------")
        print("-------------------------------------------------------------------\n")

    	# Spline interpolation
    	splineOp=interpBSpline2dModule.bSpline2d(modelStartSpline,modelStartFwi,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)

        # Nonlinear operator chain
        fwiSplineNonlinearOp=pyOpterator.ChainOperator(splineOp,nonlinearVelocityOp)

        # Linear operator chain
        fwiSplineLinearOp=pyOpterator.ChainOperator(splineOp,BornOp)

        # FWI operator
        fwiOp=pyOp.NonLinearOperator(fwiSplineNonlinearOp,fwiSplineLinearOp,BornOp.setVel)

    # Spline interpolation only
    if (dataTaper==1 and spline==1):

        print("-------------------------------------------------------------------")
        print("--------------------------- Running FWI ---------------------------")
        print("--------------- Spline interpolation + data tapering --------------")
        print("-------------------------------------------------------------------\n")



    else:

        # FWI
        fwiOp=pyOp.NonLinearOperator(nonlinearVelocityOp,BornOp,BornOp.setVel)

    ############################# Solver #######################################
    # L2-norm nonlinear problem
    invertedModel=modelStartFwiFloat.clone()
    fwiProb=Prblm.ProblemL2NonLinear(invertedModel,data,fwiOp)

    # Stopper
    stop=Stopper.BasicStopper(niter=parObject.getInt("nIter"))

    # Solver
    logFile=parObject.getString("logFile")
    invPrefix=parObject.getString("prefix")
    NLCGsolver=NLCG.NLCGsolver(stop,logger=logger(logFile))
    NLCGsolver.setDefaults(save_obj=True,save_res=True,save_grad=True,save_model=True,prefix=invPrefix,iter_sampling=1)

    # Run solver
    NLCGsolver.run(fwiProb,verbose=True)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

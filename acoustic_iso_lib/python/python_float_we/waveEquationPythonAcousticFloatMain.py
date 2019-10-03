#!/usr/bin/env python3.6
import sys
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float_we
import Mask3d 
import pyOperator as pyOp
import numpy as np
import time

if __name__ == '__main__':
    io=genericIO.pyGenericIO.ioModes(sys.argv)
    ioDef=io.getDefaultIO()
    parObject=ioDef.getParamObj()
    # Initialize operator
    modelFloat,dataFloat,slsqFloat,parObject,tempWaveEquationOp = Acoustic_iso_float_we.waveEquationOpInitFloat(sys.argv)
    timeMask=0;

    maskWidth=parObject.getInt("maskWidth",0)
	
    mask3dOp = Mask3d.mask3d(modelFloat,modelFloat,maskWidth,modelFloat.getHyper().axes[0].n-maskWidth,maskWidth,modelFloat.getHyper().axes[1].n-maskWidth,0,modelFloat.getHyper().axes[2].n-timeMask,0)
    waveEquationAcousticOp = pyOp.ChainOperator(tempWaveEquationOp,mask3dOp)

    print("*** domain and range checks *** ")
    print("* Amp - f * ")
    print("Am domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
    print("p shape: ", modelFloat.getNdArray().shape)
    print("Am range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
    print("f shape: ", dataFloat.getNdArray().shape)
      
    #run dot product
    if (parObject.getInt("dp",0)==1):
        tempWaveEquationOp.dotTest(verb=True)
        #waveEquationAcousticOp.dotTest(verb=True)

    # Forward
    if (parObject.getInt("adj",0) == 0):

        print("-------------------------------------------------------------------")
        print("--------- Running Python wave equation acoustic forward -----------")
        print("-------------------------------------------------------------------\n")

        # Check that model was provided
        modelFile=parObject.getString("model","noModelFile")
        if (modelFile == "noModelFile"):
            print("**** ERROR: User did not provide model file ****\n")
            quit()
        dataFile=parObject.getString("data","noDataFile")
        if (dataFile == "noDataFile"):
            print("**** ERROR: User did not provide data file name ****\n")
            quit()
        #modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)
        modelFloat=genericIO.defaultIO.getVector(modelFile)

        #run Nonlinear forward without wavefield saving
        waveEquationAcousticOp.forward(False,modelFloat,dataFloat)
      
        #write data to disk
        genericIO.defaultIO.writeVector(dataFile,dataFloat)


    # Adjoint
    else:

        print("-------------------------------------------------------------------")
        print("--------- Running Python wave equation acoustic adjoint -----------")
        print("-------------------------------------------------------------------\n")

        # Check that data was provided
        dataFile=parObject.getString("data","noDataFile")
        if (dataFile == "noDataFile"):
            print("**** ERROR: User did not provide data file ****\n")
            quit()
        modelFile=parObject.getString("model","noModelFile")
        if (modelFile == "noModelFile"):
            print("**** ERROR: User did not provide model file name ****\n")
            quit()
        #modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)
        dataFloat=genericIO.defaultIO.getVector(dataFile)

        #run Nonlinear forward without wavefield saving
        waveEquationAcousticOp.adjoint(False,modelFloat,dataFloat)

        #write data to disk
        genericIO.defaultIO.writeVector(dataFile,dataFloat)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

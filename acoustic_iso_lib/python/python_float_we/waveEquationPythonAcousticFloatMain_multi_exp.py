#!/usr/bin/env python3.5
import sys
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float_we
import pyOperator as pyOp
import numpy as np
import time

if __name__ == '__main__':
    parObject=genericIO.io(params=sys.argv)
    # Initialize operator
    modelFloat,dataFloat,slsqFloat,parObject,waveEquationAcousticOp = Acoustic_iso_float_we.waveEquationOpInitFloat_multi_exp(sys.argv)

    print("*** domain and range checks *** ")
    print("* Amp - f * ")
    print("Am domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
    print("p shape: ", modelFloat.getNdArray().shape)
    print("Am range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
    print("f shape: ", dataFloat.getNdArray().shape)

    #run dot product
    if (parObject.getInt("dp",0)==1):
        waveEquationAcousticOp.dotTest(verb=True)
        #waveEquationAcousticOp.dotTest(verb=True)

    # Forward
    if (parObject.getInt("adj",0) == 0):

        print("-------------------------------------------------------------------")
        print("--------- Running Python wave equation multi experimentacoustic forward -----------")
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
        modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=4)

        #run Nonlinear forward without wavefield saving
        waveEquationAcousticOp.forward(False,modelFloat,dataFloat)

        #write data to disk
        genericIO.defaultIO.writeVector(dataFile,dataFloat)


    # Adjoint
    else:

        print("-------------------------------------------------------------------")
        print("--------- Running Python wave equation multi experiment acoustic adjoint -----------")
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
        dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=4)

        #run Nonlinear forward without wavefield saving
        waveEquationAcousticOp.adjoint(False,modelFloat,dataFloat)

        #write data to disk
        genericIO.defaultIO.writeVector(dataFile,dataFloat)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

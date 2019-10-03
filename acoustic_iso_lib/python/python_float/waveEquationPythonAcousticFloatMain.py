#!/usr/bin/env python3.5
import sys
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float_we
import numpy as np
import time
import StaggerFloat

if __name__ == '__main__':
    # Initialize operator
    modelFloat,dataFloat,elasticParamFloat,parObject,waveEquationAcousticOp = Acoustic_iso_float_we.waveEquationOpInitFloat(sys.argv)

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

        domain_hyper=waveEquationAcousticOp.domain.getHyper()
        model_hyper=modelFloat.getHyper()
        range_hyper=waveEquationAcousticOp.range.getHyper()
        data_hyper=dataFloat.getHyper()

        print("*** domain and range checks *** ")
        print("* Amp - f * ")
        print("Am domain: ", waveEquationAcousticOp.getDomain().getNdArray().shape)
        print("p shape: ", modelFloat.getNdArray().shape)
        print("Am range: ", waveEquationAcousticOp.getRange().getNdArray().shape)
        print("f shape: ", dataFloat.getNdArray().shape)

        #run dot product
        if (parObject.getInt("dp",0)==1):
            waveEquationAcousticOp.dotTest(verb=True)

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

        domain_hyper=waveEquationAcousticOp.domain.getHyper()
        model_hyper=modelFloat.getHyper()
        range_hyper=waveEquationAcousticOp.range.getHyper()
        data_hyper=dataFloat.getHyper()

        #run dot product
        if (parObject.getInt("dp",0)==1):
            waveEquationAcousticOp.dotTest(verb=True)

        #run Nonlinear forward without wavefield saving
        waveEquationAcousticOp.adjoint(False,modelFloat,dataFloat)

        #write data to disk
        genericIO.defaultIO.writeVector(modelFile,modelFloat)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

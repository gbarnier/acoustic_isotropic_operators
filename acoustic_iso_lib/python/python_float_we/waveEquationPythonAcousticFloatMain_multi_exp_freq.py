#!/usr/bin/env python3.5
import sys
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float_we_freq
import pyOperator as pyOp
import numpy as np
import Mask4d
import time

if __name__ == '__main__':
    parObject=genericIO.io(params=sys.argv)
    # Initialize operator
    modelFloat,dataFloat,slsqFloat,parObject,waveEquationAcousticOpTemp,fftOp,timeFloat = Acoustic_iso_float_we_freq.waveEquationOpInitFloat_multi_exp_freq(sys.argv)
    # n1min=10
    # n1max=modelFloat.getHyper().getAxis(1).n-10
    # n2min=10
    # n2maxmodelFloat.getHyper().getAxis(2).n-10
    # n3min=0
    # n3max=modelFloat.getHyper().getAxis(3).n
    # n4min=0
    # n4max=modelFloat.getHyper().getAxis(4).n
    # mask4dOp=Mask4d.mask4d_complex(modelFloat,dataFloat,n1min,n1max,n2min,n2max,n3min,n3max,n4min,n4max,maskType=0)
    # waveEquationAcousticOp = pyOp.ChainOperator(waveEquationAcousticOpTemp,mask4dOp)
    waveEquationAcousticOp=waveEquationAcousticOpTemp

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
        print("--------- Running Python wave equation freq multi experimentacoustic forward -----------")
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

        #check if provided in time domain
        inputMode=parObject.getString("inputMode","freq")
        if (inputMode == 'time'):
            print('------ input model in time domain. converting to freq ------')
            timeFloat = genericIO.defaultIO.getVector(modelFile,ndims=4)

            fftOp.adjoint(0,modelFloat,timeFloat)
            genericIO.defaultIO.writeVector('freq_model.H',modelFloat)
        else:
            modelFloat=genericIO.defaultIO.getVector(modelFile)

        #run forward
        waveEquationAcousticOp.forward(False,modelFloat,dataFloat)

        #check if provided in time domain
        outputMode=parObject.getString("outputMode","freq")
        if (outputMode == 'time'):
            print('------ output mode is time domain. converting to time ------')
            fftOp.forward(0,dataFloat,timeFloat)
            #write data to disk
            genericIO.defaultIO.writeVector(dataFile,timeFloat)
            genericIO.defaultIO.writeVector('freq_fwd.H',dataFloat)
        else:
            genericIO.defaultIO.writeVector(dataFile,dataFloat)

    # Adjoint
    else:

        print("-------------------------------------------------------------------")
        print("--------- Running Python wave equation freq multi experiment acoustic adjoint -----------")
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

        #check if provided in time domain
        inputMode=parObject.getString("inputMode","freq")
        if (inputMode == 'time'):
            print('------ input data in time domain. converting to freq ------')
            timeFloat = genericIO.defaultIO.getVector(modelFile,ndims=4)
            fftOp.adjoint(0,dataFloat,timeFloat)
        else:
            modelFloat=genericIO.defaultIO.getVector(modelFile)

        #run Nonlinear forward without wavefield saving
        waveEquationAcousticOp.adjoint(False,modelFloat,dataFloat)

        #check if provided in time domain
        outputMode=parObject.getString("outputMode","freq")
        if (outputMode == 'time'):
            print('------ output mode is time domain. converting to time ------')
            fftOp.forward(0,modelFloat,timeFloat)
            #write data to disk
            genericIO.defaultIO.writeVector(modelFile,timeFloat)
            genericIO.defaultIO.writeVector("freq_adj.H",modelFloat)
        else:
            genericIO.defaultIO.writeVector(modelFile,modelFloat)

    print("-------------------------------------------------------------------")
    print("--------------------------- All done ------------------------------")
    print("-------------------------------------------------------------------\n")

#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_double
import numpy as np
import time
import sys

if __name__ == '__main__':

    # Seismic operator object initialization
    modelDouble,dataDouble,velDouble,parObject,sourcesVector,sourcesSignalsVector,receiversVector,wemvaDataDouble=Acoustic_iso_double.wemvaExtOpInitDouble(sys.argv)

    # Construct Born operator object
    wemvaExtOp=Acoustic_iso_double.wemvaExtShotsGpu(modelDouble,dataDouble,velDouble,parObject.param,sourcesVector,sourcesSignalsVector,receiversVector,wemvaDataDouble)

    # Launch forward modeling
    if (parObject.getInt("adj", 0) == 0):

        print("-------------------------------------------------------------------")
        print("--------------- Running Python wemva extended forward -------------")
        print("-------------------- Double precision Python code -----------------")
        print("-------------------------------------------------------------------\n")

        # Check that model was provided
        modelFile=parObject.getString("model","noModelFile")
        if (modelFile == "noModelFile"):
            print("**** ERROR: User did not provide model file ****\n")
            quit()

        # Read model
        modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=2)
        modelDMat=modelDouble.getNdArray()
        modelSMat=modelFloat.getNdArray()
        modelDMat[:]=modelSMat

        # Apply forward
        wemvaExtOp.forward(False,modelDouble,dataDouble)

        # Write data
        dataFloat=SepVector.getSepVector(dataDouble.getHyper(),storage="dataFloat")
        dataFloatNp=dataFloat.getNdArray()
        dataDoubleNp=dataDouble.getNdArray()
        dataFloatNp[:]=dataDoubleNp
        dataFile=parObject.getString("data","noDataFile")
        if (dataFile == "noDataFile"):
            print("**** ERROR: User did not provide data file name ****\n")
            quit()
        genericIO.defaultIO.writeVector(dataFile,dataFloat)

        print("-------------------------------------------------------------------")
        print("--------------------------- All done ------------------------------")
        print("-------------------------------------------------------------------\n")

    # Launch adjoint modeling
    else:

        print("-------------------------------------------------------------------")
        print("---------------- Running Python wemva extended adjoint ------------")
        print("-------------------- Double precision Python code -----------------")
        print("-------------------------------------------------------------------\n")

        # Check that data was provided
        dataFile=parObject.getString("data","noDataFile")
        if (dataFile == "noDataFile"):
            print("**** ERROR: User did not provide data file ****\n")
            quit()

        # Read data
        dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)
        dataFloatNp=dataFloat.getNdArray()
        dataDoubleNp=dataDouble.getNdArray()
        dataDoubleNp[:]=dataFloatNp

        # Apply adjoint
        wemvaExtOp.adjoint(False,modelDouble,dataDouble)

        # Write model
        modelFloat=SepVector.getSepVector(modelDouble.getHyper(),storage="dataFloat")
        modelFloatNp=modelFloat.getNdArray()
        modelDoubleNp=modelDouble.getNdArray()
        modelFloatNp[:]=modelDoubleNp
        modelFile=parObject.getString("model","noModelFile")
        if (modelFile == "noModelFile"):
            print("**** ERROR: User did not provide model file name ****\n")
            quit()
        genericIO.defaultIO.writeVector(modelFile,modelFloat)

        print("-------------------------------------------------------------------")
        print("--------------------------- All done ------------------------------")
        print("-------------------------------------------------------------------\n")

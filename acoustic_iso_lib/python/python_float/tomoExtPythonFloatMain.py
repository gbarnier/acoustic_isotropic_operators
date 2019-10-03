#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import numpy as np
import time
import sys

if __name__ == '__main__':

    # Seismic operator object initialization
    modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,reflectivityFloat=Acoustic_iso_float.tomoExtOpInitFloat(sys.argv)

    # Construct Born operator object
    tomoExtOp=Acoustic_iso_float.tomoExtShotsGpu(modelFloat,dataFloat,velFloat,parObject.param,sourcesVector,sourcesSignalsVector,receiversVector,reflectivityFloat)

    # Forward
    if (parObject.getInt("adj", 0) == 0):

        print("-------------------------------------------------------------------")
        print("--------------- Running Python tomo extended forward --------------")
        print("-------------------- Single precision Python code -----------------")
        print("-------------------------------------------------------------------\n")

        # Check that model was provided
        modelFile=parObject.getString("model","noModelFile")
        if (modelFile == "noModelFile"):
            print("**** ERROR: User did not provide model file ****\n")
            quit()

        # Read model
        modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=2)

        # Apply forward
        tomoExtOp.forward(False,modelFloat,dataFloat)

        # Write data
        dataFile=parObject.getString("data","noDataFile")
        if (dataFile == "noDataFile"):
            print("**** ERROR: User did not provide data file name ****\n")
            quit()
        genericIO.defaultIO.writeVector(dataFile,dataFloat)

        print("-------------------------------------------------------------------")
        print("--------------------------- All done ------------------------------")
        print("-------------------------------------------------------------------\n")

    # Adjoint
    else:

        print("-------------------------------------------------------------------")
        print("---------------- Running Python tomo extended adjoint -------------")
        print("-------------------- Single precision Python code -----------------")
        print("-------------------------------------------------------------------\n")

        # Check that data was provided
        dataFile=parObject.getString("data","noDataFile")
        if (dataFile == "noDataFile"):
            print("**** ERROR: User did not provide data file ****\n")
            quit()

        # Read data
        dataFloat=genericIO.defaultIO.getVector(dataFile,ndims=3)

        # Apply adjoint
        tomoExtOp.adjoint(False,modelFloat,dataFloat)

        # Write model
        modelFile=parObject.getString("model","noModelFile")
        if (modelFile == "noModelFile"):
            print("**** ERROR: User did not provide model file name ****\n")
            quit()
        genericIO.defaultIO.writeVector(modelFile,modelFloat)

        print("-------------------------------------------------------------------")
        print("--------------------------- All done ------------------------------")
        print("-------------------------------------------------------------------\n")

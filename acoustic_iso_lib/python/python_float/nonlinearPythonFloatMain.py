#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_float
import numpy as np
import time
import sys

if __name__ == '__main__':

    # Initialize operator
    modelFloat,dataFloat,velFloat,parObject,sourcesVector,receiversVector=Acoustic_iso_float.nonlinearOpInitFloat(sys.argv)

    # Construct nonlinear operator object
    nonlinearOp=Acoustic_iso_float.nonlinearPropShotsGpu(modelFloat,dataFloat,velFloat,parObject,sourcesVector,receiversVector)

    # Forward
    if (parObject.getInt("adj",0) == 0):

        print("-------------------------------------------------------------------")
        print("------------------ Running Python nonlinear forward ---------------")
        print("-------------------- Single precision Python code -----------------")
        print("-------------------------------------------------------------------\n")

        # Check that model was provided
        modelFile=parObject.getString("model","noModelFile")
        if (modelFile == "noModelFile"):
            print("**** ERROR: User did not provide model file ****\n")
            quit()

        # Read model
        modelFloat=genericIO.defaultIO.getVector(modelFile,ndims=3)

        # Apply forward
        nonlinearOp.forward(False,modelFloat,dataFloat)

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
        print("----------------- Running Python nonlinear adjoint ----------------")
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
        nonlinearOp.adjoint(False,modelFloat,dataFloat)

        # Write model
        modelFile=parObject.getString("model","noModelFile")
        if (modelFile == "noModelFile"):
            print("**** ERROR: User did not provide model file name ****\n")
            quit()
        genericIO.defaultIO.writeVector(modelFile,modelFloat)

        print("-------------------------------------------------------------------")
        print("--------------------------- All done ------------------------------")
        print("-------------------------------------------------------------------\n")

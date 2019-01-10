#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import Acoustic_iso_double
import numpy as np
import time
import sys

if __name__ == '__main__':

    # Initialize operator
    modelDouble, dataDouble, velDouble, parObject, sourcesVector, receiversVector = Acoustic_iso_double.nonlinearOpInitDouble(sys.argv)

    # Construct nonlinear operator object
    nonlinearOp = Acoustic_iso_double.nonlinearPropShotsGpu(modelDouble, dataDouble, velDouble, parObject, sourcesVector, receiversVector)

    # Launch forward modeling
    if (parObject.getInt("adj", 0) == 0):

        print("-------------------------------------------------------------------")
        print("----------------------- Running nonlinear forward -----------------")
        print("-------------------------------------------------------------------")
        nonlinearOp.forward(False, modelDouble, dataDouble) # Run forward

        # Write data
        data=SepVector.getSepVector(dataDouble.getHyper(), storage="dataFloat")
        dataNp=data.getNdArray()
        dataDoubleNp=dataDouble.getNdArray()
        dataNp[:]=dataDoubleNp
        dataFile=parObject.getString("data")
        genericIO.defaultIO.writeVector(dataFile, data)

    # Launch adjoint modeling
    else:

        print("-------------------------------------------------------------------")
        print("----------------------- Running nonlinear adjoint -----------------")
        print("-------------------------------------------------------------------")
        nonlinearOp.adjoint(False, modelDouble, dataDouble) # Run adjoint

        # Write model
        model=SepVector.getSepVector(modelDouble.getHyper(), storage="dataFloat")
        modelNp=model.getNdArray()
        modelDoubleNp=modelDouble.getNdArray()
        modelNp[:]=modelDoubleNp
        modelFile = parObject.getString("model")
        genericIO.defaultIO.writeVector(modelFile, model)

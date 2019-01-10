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
    modelDouble, dataDouble, velDouble, parObject, sourcesVector, sourcesSignalsVector, receiversVector = Acoustic_iso_double.BornExtOpInitDouble(sys.argv)

    # Construct Born operator object
    BornExtOp = Acoustic_iso_double.BornExtShotsGpu(modelDouble, dataDouble, velDouble, parObject, sourcesVector, sourcesSignalsVector, receiversVector)

    # Launch forward modeling
    if (parObject.getInt("adj", 0) == 0):

        print("-------------------------------------------------------------------")
        print("------------------- Running Born extended forward -----------------")
        print("-------------------------------------------------------------------")
        BornExtOp.forward(False, modelDouble, dataDouble)

        # Write data
        data=SepVector.getSepVector(dataDouble.getHyper(), storage="dataFloat")
        dataNp=data.getNdArray()
        dataDoubleNp=dataDouble.getNdArray()
        dataNp[:]=dataDoubleNp
        dataFile = parObject.getString("data")
        genericIO.defaultIO.writeVector(dataFile, data);

    # Launch adjoint modeling
    else:

        print("-------------------------------------------------------------------")
        print("------------------- Running Born extended adjoint -----------------")
        print("-------------------------------------------------------------------")
        BornExtOp.adjoint(False, modelDouble, dataDouble)

        # Write model
        model=SepVector.getSepVector(modelDouble.getHyper(), storage="dataFloat")
        modelNp=model.getNdArray()
        modelDoubleNp=modelDouble.getNdArray()
        modelNp[:]=modelDoubleNp
        modelFile = parObject.getString("model")
        genericIO.defaultIO.writeVector(modelFile, model)

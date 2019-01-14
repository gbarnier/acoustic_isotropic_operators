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
    modelDouble, dataDouble, velDouble, parObject, sourcesVector, sourcesSignalsVector, receiversVector, wemvaDataDouble = Acoustic_iso_double.wemvaExtOpInitDouble(sys.argv)

    # Construct Born operator object
    wemvaExtOp = Acoustic_iso_double.wemvaExtShotsGpu(modelDouble, dataDouble, velDouble, parObject, sourcesVector, sourcesSignalsVector, receiversVector, wemvaDataDouble)

    # Launch forward modeling
    if (parObject.getInt("adj", 0) == 0):

        print("-------------------------------------------------------------------")
        print("------------------- Running tomo extended forward -----------------")
        print("-------------------------------------------------------------------")
        wemvaExtOp.forward(False, modelDouble, dataDouble)

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
        print("------------------- Running tomo extended adjoint -----------------")
        print("-------------------------------------------------------------------")
        wemvaExtOp.adjoint(False, modelDouble, dataDouble)

        # Write model
        model=SepVector.getSepVector(modelDouble.getHyper(), storage="dataFloat")
        modelNp=model.getNdArray()
        modelDoubleNp=modelDouble.getNdArray()
        modelNp[:]=modelDoubleNp
        modelFile = parObject.getString("model")
        genericIO.defaultIO.writeVector(modelFile, model)

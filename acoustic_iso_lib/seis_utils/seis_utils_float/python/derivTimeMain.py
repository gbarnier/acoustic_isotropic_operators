#!/usr/bin/env python3
"""
First-order time derivative operator

USAGE EXAMPLE:
	derivTimeMain.py input= output=

INPUT PARAMETERS:
	input = [no default] - string; Header file containing the data on which the time derivative will be computed (along n1)

	output = [no default] - string; Header file name in which output will be written
"""
import sys
import genericIO
import numpy as np

if __name__ == '__main__':
	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Get input and output filenames
	inputFile=parObject.getString("input","noInput")
	if (inputFile == "noInput"):
		raise IOError("**** ERROR: User did not provide input file ****\n")
	outputFile=parObject.getString("output","noOutput")
	if (outputFile == "noOutput"):
		raise IOError("**** ERROR: User did not provide output filename ****\n")
	input=genericIO.defaultIO.getVector(inputFile)
	nDims = input.getHyper().getNdim()
	if(nDims != 1):
		raise ValueError("ERROR! derivTimeMain currently supports only one dimensional files")

	#Getting time axis info
	timeAx = input.getHyper().getAxis(1)
	nt = timeAx.n
	dt = timeAx.d
	coef = 1.0 / (2.0 * dt)

	#Computing derivative
	output = input.clone()
	output.zero()
	inNd = input.getNdArray()
	outNd = output.getNdArray()
	#Boundary conditions
	outNd[0] = inNd[1]*coef
	outNd[-1] = -inNd[nt-2]*coef
	for it in range(1,nt-1):
		outNd[it] = coef*(inNd[it+1]-inNd[it-1])

	#Writing output vector
	genericIO.defaultIO.writeVector(outputFile,output)

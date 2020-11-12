#!/usr/bin/env python3
"""
interpPSF.py model= zPadMinus= zPadPlus= xPadMinus= xPadPlus= fat= nPSF_= interpPSF=

Linear interpolation of the point-spread functions
"""
import numpy as np
import pyVector as pyVec
import sys
import genericIO
from scipy.interpolate import RegularGridInterpolator


if __name__ == "__main__":
	#Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

	parObject=genericIO.io(params=sys.argv)
	# Read input and output files
	modelFile=parObject.getString("model")
	interpFile=parObject.getString("interpPSF")
	model = genericIO.defaultIO.getVector(modelFile, ndims=3)
	modelNd = model.getNdArray()
	nz = model.getHyper().getAxis(1).n
	nx = model.getHyper().getAxis(2).n
	nExt = model.getHyper().getAxis(3).n
	# Getting padding to place PSF in the modeling area
	zPadMinus = parObject.getInt("zPadMinus")
	zPadPlus = parObject.getInt("zPadPlus")
	xPadMinus = parObject.getInt("xPadMinus")
	xPadPlus = parObject.getInt("xPadPlus")
	fat = parObject.getInt("fat")
	# Getting number of PSFs on each axis
	nPSF1 = parObject.getInt("nPSF1")
	nPSF2 = parObject.getInt("nPSF2")
	nPSF3 = parObject.getInt("nPSF3",1)

	if nPSF3 != 1 and nExt == 1:
		raiseValueError("ERROR! nPSF3 must be 1 for non-extended images")
	if nPSF3 == 1 and nExt != 1:
		raiseValueError("ERROR! nPSF3 must be different than 1 for extended images")
	if nPSF1 <= 0 or nPSF2 <= 0 or nPSF3 <= 0:
		raiseValueError("ERROR! nPSF must be positive")

	# Computing non-padded model size
	nz_nopad = nz - 2*fat - zPadMinus - zPadPlus
	nx_nopad = nx - 2*fat - xPadMinus - xPadPlus
	PSF1_pos = np.linspace(zPadMinus+fat, zPadMinus+fat+nz_nopad, nPSF1).astype(np.int)
	PSF2_pos = np.linspace(xPadMinus+fat, xPadMinus+fat+nx_nopad, nPSF2).astype(np.int)
	if nPSF3 != 1:
		PSF3_pos = np.linspace(0, nExt-1, nPSF3).astype(np.int)
		ee,xx,zz = np.meshgrid(PSF3_pos,PSF2_pos,PSF1_pos,indexing='ij',sparse=True)
		PSF_values = modelNd[ee,xx,zz]
		points = (PSF3_pos,PSF2_pos,PSF1_pos)
		ee1,xx1,zz1 = np.meshgrid(np.arange(nExt),np.arange(nx),np.arange(nz),indexing='ij',sparse=True)
		modelMesh = (ee1,xx1,zz1)
	else:
		# Filling positions of the PSF with ones
		xx,zz = np.meshgrid(PSF2_pos,PSF1_pos,indexing='ij',sparse=True)
		PSF_values = modelNd[0,xx,zz]
		points = (PSF2_pos,PSF1_pos)
		xx1,zz1 = np.meshgrid(np.arange(nx),np.arange(nz),indexing='ij',sparse=True)
		modelMesh = (xx1,zz1)

	f = RegularGridInterpolator(points, PSF_values, bounds_error=False)
	modelNd[:] = f(modelMesh)
	# Filling the padding
	modelNd[:,0:xPadMinus+fat,:] = np.reshape(modelNd[:,xPadMinus+fat,:],(nExt,1,nz))
	modelNd[:,-(xPadPlus+fat):,:] = np.reshape(modelNd[:,-(xPadPlus+fat),:],(nExt,1,nz))
	modelNd[:,:,0:zPadMinus+fat] = np.reshape(modelNd[:,:,zPadMinus+fat],(nExt,nx,1))
	modelNd[:,:,-(zPadPlus+fat):] = np.reshape(modelNd[:,:,-(zPadPlus+fat)],(nExt,nx,1))
	model.writeVec(interpFile)

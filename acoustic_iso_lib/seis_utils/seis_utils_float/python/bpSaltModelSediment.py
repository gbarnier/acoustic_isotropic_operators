#!/usr/bin/env python3
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

if __name__ == '__main__':

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Read true model
	modelFile=parObject.getString("model")
	model=genericIO.defaultIO.getVector(modelFile,ndims=2)
	modelNp=model.getNdArray()

	############################################################################
	######################### Find water bottom index ##########################
	############################################################################
	# Get dimensions
	nz=model.getHyper().axes[0].n
	oz=model.getHyper().axes[0].o
	dz=model.getHyper().axes[0].d
	nx=model.getHyper().axes[1].n
	ox=model.getHyper().axes[1].o
	dx=model.getHyper().axes[1].d
	modelNp=model.getNdArray()
	xAxis=model.getHyper().axes[1]

	# Create water bottom INDEX array
	wbIndexHyper=Hypercube.hypercube(axes=[xAxis])
	wbIndex=SepVector.getSepVector(wbIndexHyper)
	wbIndexNp=wbIndex.getNdArray()

	# Create water bottom DEPTH array
	wbDepth=SepVector.getSepVector(wbIndex.getHyper())
	wbDepthNp=wbDepth.getNdArray()

	# Get water velocity (assumed to be constant)
	waterVelocity=modelNp[0][0]
	print("Minimum model velocity",np.amin(modelNp))
	print("Maximum model velocity",np.amax(modelNp))
	print("Water velocity",waterVelocity)

	# Compute water bottom index and depth
	for ix in range(nx):
		iz=0
		while (modelNp[ix][iz] == waterVelocity):
			wbIndexNp[ix]=iz
			wbDepthNp[ix]=(iz-1)*dz+oz
			iz=iz+1

	# write water bottom index
	wbIndexFile=parObject.getString("wbIndex")
	genericIO.defaultIO.writeVector(wbIndexFile,wbIndex)

	# write water bottom index
	wbDepthFile=parObject.getString("wbDepth")
	genericIO.defaultIO.writeVector(wbDepthFile,wbDepth)

	############################################################################
	######################### Create salt mask #################################
	############################################################################
	# Coordinates of box containing the salt body
	zMaxSalt=7.000 # [km]
	# zMaxCarapace=3.500 # [km]
	xMinSalt=0 # [km]
	xMaxSalt=22.500 # [km]
	velMinSalt=4.440 # [m/s]
	velMaxSalt=4.600 # [m/s]
	# velMinCarapace=4000 # [km]

	# Salt mask
	maskSalt=SepVector.getSepVector(model.getHyper())
	maskSaltNp=maskSalt.getNdArray()
	maskSalt.set(0.0)

	# Fill salt mask
	for ix in range(nx):
		x=ox+(ix-1)*dx
		if (x>xMinSalt and x<xMaxSalt): # Check if you are inside the box
			zWaterBottom=wbDepthNp[ix] # Compute water bottom depth
			for iz in range(nz):
				z=oz+(iz-1)*dz
				if (z>zWaterBottom and z<zMaxSalt): # Check if you are inside the box
					# # Case 1: you are in the depth range where there are carapaces
					# if (z<zMaxCarapace and modelNp[ix][iz]>velMinCarapace):
					# 	maskSaltNp[ix][iz]=1
					# Case 2: you are below the depth range where there are carapaces
					if (modelNp[ix][iz]>velMinSalt and modelNp[ix][iz]<velMaxSalt):
						maskSaltNp[ix][iz]=1

	# Write mask
	maskSaltFile=parObject.getString("saltMask")
	genericIO.defaultIO.writeVector(maskSaltFile,maskSalt)

	print("Done computing salt mask")

	############################################################################
	################## Fill in water layer with sediment velocity ##############
	############################################################################
	# Fill in water layer by padding
	velFill=model.clone()
	velFillNp=velFill.getNdArray()

	for ix in range(nx):
		index=int(wbIndexNp[ix]+1)
		for iz in range(index):
			velFillNp[ix][iz]=modelNp[ix][index+1]

	velFillFile=parObject.getString("velFill")
	genericIO.defaultIO.writeVector(velFillFile,velFill)

	print("Done filling in water layer")

	############################################################################
	################## Replace salt velocity by local average ##################
	############################################################################
	# Initial size of the averaging box
	elementCountMinSide=parObject.getInt("elementCountMinSide",10)
	elementCountMin=4*elementCountMinSide

	# Allocate output
	data=velFill.clone()
	dataNp=data.getNdArray()

	# Number of points used in the averaging
	elementCount=velFill.clone()
	elementCount.set(0.0) # Set the element count to zero for all points
	elementCountNp=elementCount.getNdArray()

	# Interpolated points
	interp=velFill.clone() # Flag that indicates if the velocity value for that pixel has been computed
	interp.set(1.0)
	interpNp=interp.getNdArray()

	# Make sure interp is set to zero
	print("interpMax",np.amax(interpNp))
	print("interpMin",np.amin(interpNp))

	# Loop over horizontal position
	for ix in range(nx):
		print("ix=",ix)

		# Loop over vertical position
		for iz in range(nz):

			# Only interpolate points in the salt
			if (maskSaltNp[ix][iz] == 1):

				velSum=0 # Cumulative velocity sum

				# Search left
				countLeft=0
				ixProp=ix
				while (countLeft<elementCountMin and ixProp>0):
					ixProp=ixProp-1
					if (maskSaltNp[ixProp][iz] != 1): # Check if the proposed point is not salt
						velSum=velSum+velFillNp[ixProp][iz]
						countLeft=countLeft+1
						elementCountNp[ix][iz]=elementCountNp[ix][iz]+1

				# Search left
				countRight=0
				ixProp=ix
				while (countRight<elementCountMin and ixProp<nx-1):
					ixProp=ixProp+1
					if (maskSaltNp[ixProp][iz] != 1): # Check if the proposed point is not salt
						velSum=velSum+velFillNp[ixProp][iz]
						countRight=countRight+1
						elementCountNp[ix][iz]=elementCountNp[ix][iz]+1

				# Search Top
				countTop=0
				izProp=iz
				while (countTop<elementCountMin and izProp>0):
					izProp=izProp-1
					if (maskSaltNp[ix][izProp] != 1): # Check if the proposed point is not salt
						velSum=velSum+velFillNp[ix][izProp]
						countTop=countTop+1
						elementCountNp[ix][iz]=elementCountNp[ix][iz]+1

				# Search Bottom
				countBottom=0
				izProp=iz
				while (countBottom<elementCountMin and izProp<nz-1):
					izProp=izProp+1
					if (maskSaltNp[ix][izProp] != 1): # Check if the proposed point is not salt
						velSum=velSum+velFillNp[ix][izProp]
						countBottom=countBottom+1
						elementCountNp[ix][iz]=elementCountNp[ix][iz]+1

				# If we didn't manage to grab enough points on each side
				if (elementCountNp[ix][iz] != elementCountMin):
					interpNp[ix][iz]=0
					print("iz=",iz)
					print("ix=",ix)
					print("nb of points=",elementCountNp[ix][iz])


				# Make sure we have at least one point
				if (elementCountNp[ix][iz]>0):
					dataNp[ix][iz]=velSum/elementCountNp[ix][iz]
				else:
					dataNp[ix][iz]=-1

	# Put the water velocity back inside the water layer
	# for ix in range(nx):
	# 	index=int(wbIndexNp[ix]+1)
	# 	for iz in range(index):
	# 		dataNp[ix][iz]=modelNp[ix][iz]

	# Write interpolated "smooth" model
	dataFile=parObject.getString("data")
	genericIO.defaultIO.writeVector(dataFile,data)

	# Write element count array
	elementCountFile=parObject.getString("elementCount")
	genericIO.defaultIO.writeVector(elementCountFile,elementCount)

	# Write interpreted points array
	interpFile=parObject.getString("interp")
	genericIO.defaultIO.writeVector(interpFile,interp)

	print("All done")

#!/usr/bin/env python3
"""


USAGE EXAMPLE:
	off2angMain.py off_img= ang_img= nh= oh= dh= ng= og= dg= adj=

INPUT PARAMETERS:
"""

import genericIO
import SepVector
import Hypercube
from off2angModule import off2ang2D
import numpy as np
import sys


if __name__ == '__main__':
	# Printing documentation if no arguments were provided
	if(len(sys.argv) == 1):
		print(__doc__)
		quit(0)

	# Getting parameter object
	parObject = genericIO.io(params=sys.argv)

	# Other parameters
	adj = parObject.getBool("adj",0)

	# Getting axes' info
	if adj:
		# subsurface-offset axis (Adjoint)
		nh = parObject.getInt("nh")
		dh = parObject.getFloat("dh")
		oh = parObject.getFloat("oh")
	else:
		# angle axis (Forward)
		ng = parObject.getInt("ng")
		dg = parObject.getFloat("dg")
		og = parObject.getFloat("og")


	# Check whether file names were provided or not
	off_img_file=parObject.getString("off_img","None")
	if off_img_file == "None":
		raise ValueError("**** ERROR: User did not provide subsurface-offset-domain image file ****")

	ang_img_file=parObject.getString("ang_img","None")
	if ang_img_file == "None":
		raise ValueError("**** ERROR: User did not provide angle-domain image file ****")

	# Applying forward
	if adj == 0:
		# Read offset-domain image
		ODCIGs = genericIO.defaultIO.getVector(off_img_file)
		# Getting axis
		z_axis = ODCIGs.getHyper().getAxis(1)
		x_axis = ODCIGs.getHyper().getAxis(2)
		h_axis = ODCIGs.getHyper().getAxis(3)
		g_axis = Hypercube.axis(n=ng,o=og,d=dg,label="\F9 g \F-1 [deg]")
		ADCIGs = SepVector.getSepVector(Hypercube.hypercube(axes=[z_axis,x_axis,g_axis]))

		# Constructing operator
		off2angOp = off2ang2D(ODCIGs,ADCIGs,z_axis.o,z_axis.d,h_axis.o,h_axis.d,g_axis.o,g_axis.d)

		# Applying transformation
		off2angOp.forward(False,ODCIGs,ADCIGs)
		# Writing result
		ADCIGs.writeVec(ang_img_file)

	# Applying adjoint
	else:
		# Read offset-domain image
		ADCIGs = genericIO.defaultIO.getVector(ang_img_file)
		# Getting axis
		z_axis = ADCIGs.getHyper().getAxis(1)
		x_axis = ADCIGs.getHyper().getAxis(2)
		g_axis = ADCIGs.getHyper().getAxis(3)
		h_axis = Hypercube.axis(n=nh,o=oh,d=dh,label="suburface offset")
		ODCIGs = SepVector.getSepVector(Hypercube.hypercube(axes=[z_axis,x_axis,h_axis]))

		# Constructing operator
		off2angOp = off2ang2D(ODCIGs,ADCIGs,z_axis.o,z_axis.d,h_axis.o,h_axis.d,g_axis.o,g_axis.d)
		# Applying transformation
		off2angOp.adjoint(False,ODCIGs,ADCIGs)
		# Writing result
		ODCIGs.writeVec(off_img_file)

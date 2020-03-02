#!/usr/bin/env python3
"""


USAGE EXAMPLE:
	off2angMain.py off_img= ang_img= nh= oh= dh= ng= og= dg= adj=1 p_inv=1 anti_alias=1

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
	adj = parObject.getBool("adj",1)
	p_inv = parObject.getBool("p_inv",1)
	anti_alias = parObject.getBool("anti_alias",1)
	dp_test = parObject.getBool("dp_test",0)

	# Getting axes' info
	if adj:
		# angle axis (Forward)
		ng = parObject.getInt("ng")
		dg = parObject.getFloat("dg")
		og = parObject.getFloat("og")
	else:
		# subsurface-offset axis (Adjoint)
		nh = parObject.getInt("nh")
		dh = parObject.getFloat("dh")
		oh = parObject.getFloat("oh")


	# Check whether file names were provided or not
	off_img_file=parObject.getString("off_img","None")
	if off_img_file == "None":
		raise ValueError("**** ERROR: User did not provide subsurface-offset-domain image file ****")

	ang_img_file=parObject.getString("ang_img","None")
	if ang_img_file == "None":
		raise ValueError("**** ERROR: User did not provide angle-domain image file ****")

	# Applying forward
	if not adj:
		# Read offset-domain image
		ADCIGs = genericIO.defaultIO.getVector(ang_img_file,ndims=3)
		# Getting axis
		z_axis = ADCIGs.getHyper().getAxis(1)
		x_axis = ADCIGs.getHyper().getAxis(2)
		g_axis = ADCIGs.getHyper().getAxis(3)
		h_axis = Hypercube.axis(n=nh,o=oh,d=dh,label="suburface offset")
		ODCIGs = SepVector.getSepVector(Hypercube.hypercube(axes=[z_axis,x_axis,h_axis]))

		# Constructing operator
		off2angOp = off2ang2D(ADCIGs,ODCIGs,z_axis.o,z_axis.d,h_axis.o,h_axis.d,g_axis.o,g_axis.d,p_inv,anti_alias)
		if dp_test:
			# Dot-product test if requested
			off2angOp.dotTest(True)
			quit()

		# Applying transformation
		off2angOp.forward(False,ADCIGs,ODCIGs)
		# Writing result
		ODCIGs.writeVec(off_img_file)

	# Applying adjoint
	else:
		# Read offset-domain image
		ODCIGs = genericIO.defaultIO.getVector(off_img_file,ndims=3)
		# Getting axis
		z_axis = ODCIGs.getHyper().getAxis(1)
		x_axis = ODCIGs.getHyper().getAxis(2)
		h_axis = ODCIGs.getHyper().getAxis(3)
		g_axis = Hypercube.axis(n=ng,o=og,d=dg,label="\F9 g \F-1 [deg]")
		ADCIGs = SepVector.getSepVector(Hypercube.hypercube(axes=[z_axis,x_axis,g_axis]))

		# Constructing operator
		off2angOp = off2ang2D(ADCIGs,ODCIGs,z_axis.o,z_axis.d,h_axis.o,h_axis.d,g_axis.o,g_axis.d,p_inv,anti_alias)
		if dp_test:
			# Dot-product test if requested
			off2angOp.dotTest(True)
			quit()

		# Applying transformation
		off2angOp.adjoint(False,ADCIGs,ODCIGs)
		# Writing result
		ADCIGs.writeVec(ang_img_file)

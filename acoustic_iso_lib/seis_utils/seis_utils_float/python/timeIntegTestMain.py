#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import timeIntegModule
import matplotlib.pyplot as plt
import sys
import time

if __name__ == '__main__':

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(sys.argv)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Initialize and instanciate time integration operator
	ots=0.0
	# dts=parObject.getFloat("dts")
	nts=parObject.getInt("nts")
	nOscillation=5.0
	dts=2.0*np.pi*nOscillation/nts

	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	dummy1=Hypercube.axis(n=1,o=0.0,d=1.0)
	dummy2=Hypercube.axis(n=1,o=0.0,d=1.0)

	################################ True signal ###############################
	f1=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,dummy1,dummy2]))
	timeIntegOp=timeIntegModule.timeInteg(f1,f1,dts)
	f1Nd=f1.getNdArray()
	for its in range(nts):
		# Sine
		t=ots+its*dts
		f1Nd[0][0][its]=np.sin(t)

	############################ Analytical integration ########################
	# First integration
	fInt1A=f1.clone()
	fInt1ANd=fInt1A.getNdArray()
	for its in range(nts):
		t=ots+its*dts
		fInt1ANd[0][0][its]=-np.cos(t)+1.0

	# Second integration
	fInt2A=f1.clone()
	fInt2ANd=fInt2A.getNdArray()
	for its in range(nts):
		t=ots+its*dts
		fInt2ANd[0][0][its]=-np.sin(t)+t

	# Third integration
	fInt3A=f1.clone()
	fInt3ANd=fInt3A.getNdArray()
	for its in range(nts):
		t=ots+its*dts
		fInt3ANd[0][0][its]=np.cos(t)+0.5*t*t-1.0

	########################### Numerical integration ##########################
	# First integral
	fInt1N=f1.clone()
	timeIntegOp.forward(False,f1,fInt1N)

	# Second integral
	fInt2N=f1.clone()
	timeIntegOp.forward(False,fInt1N,fInt2N)

	# Third integral
	fInt3N=f1.clone()
	timeIntegOp.forward(False,fInt2N,fInt3N)

	########################### Write outputs ##################################
	# Write outputs
	f1File=parObject.getString("f1")
	fInt1AFile=parObject.getString("fInt1A")
	fInt2AFile=parObject.getString("fInt2A")
	fInt3AFile=parObject.getString("fInt3A")

	fInt1NFile=parObject.getString("fInt1N")
	fInt2NFile=parObject.getString("fInt2N")
	fInt3NFile=parObject.getString("fInt3N")

	genericIO.defaultIO.writeVector(f1File,f1)
	genericIO.defaultIO.writeVector(fInt1AFile,fInt1A)
	genericIO.defaultIO.writeVector(fInt2AFile,fInt2A)
	genericIO.defaultIO.writeVector(fInt3AFile,fInt3A)

	genericIO.defaultIO.writeVector(fInt1NFile,fInt1N)
	genericIO.defaultIO.writeVector(fInt2NFile,fInt2N)
	genericIO.defaultIO.writeVector(fInt3NFile,fInt3N)

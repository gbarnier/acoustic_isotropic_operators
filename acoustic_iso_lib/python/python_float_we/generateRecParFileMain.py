#!/usr/bin/env python3.5
import genericIO
import SepVector
import Hypercube
import numpy as np
import time
import sys
import os


# Template for linearized waveform inversion workflow
if __name__ == '__main__':

	# io stuff
	parObject=genericIO.io(params=sys.argv)
	# Reads source or receiver x,y,z,exp locations from parFile

	recParFile = parObject.getString("recParFile","None")
	nExp = parObject.getInt("nExp",0)
	nzReceiver=parObject.getInt("nzReceiver",0)
	dzReceiver=parObject.getInt("dzReceiver",0)
	ozReceiver=parObject.getInt("ozReceiver",0)
	nxReceiver=parObject.getInt("nxReceiver",0)
	dxReceiver=parObject.getInt("dxReceiver",0)
	oxReceiver=parObject.getInt("oxReceiver",0)

	with open(recParFile,"w") as fid:
		for iExp in np.arange(nExp):
			for izRec in np.arange(nzReceiver):
				iz = ozReceiver + dzReceiver * izRec
				for ixRec in np.arange(nxReceiver):
					ix = oxReceiver + dxReceiver * ixRec
					lineCur = str(ix) + ' ' + str(iz) + ' ' + str(iExp) + '\n'
					fid.write(lineCur)

#!/usr/bin/env python3.5

import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np
import math

class fft_wfld(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range):
		self.setDomainRange(domain,range)
		self.nt=range.getHyper().getAxis(3).n
		# self.nw=range.getHyper().getAxis(3).n
		# self.dw = range.getHyper().getAxis(3).d
		return

	# p(z,x,w) -> p(z,x,t)
	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add): data.zero()

		model_nd = model.getNdArray()
		data_nd = data.getNdArray()

		data_nd[:] += np.fft.irfft(model_nd, axis=0 )*math.sqrt(self.nt)

		return

	# p(z,x,t) -> p(z,x,w)
	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add): model.zero()

		model_nd = model.getNdArray()
		data_nd = data.getNdArray()
		model_nd[:] += np.fft.rfft(data_nd, axis=0)/math.sqrt(self.nt)

		return

#!/usr/bin/env python3

import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np
import math

class fft_wfld(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,axis=0):
		self.setDomainRange(domain,range)
		self.nt=range.getHyper().getAxis(range.getHyper().getNdim()-axis).n
		self.axis=axis
		print('self.axis:',self.axis)
		print('nt:',self.nt)
		print('size of chosen "axis": '+str(domain.getNdArray().shape[axis]))
		# self.dw = range.getHyper().getAxis(3).d
		return

	# p(z,x,w) -> p(z,x,t)
	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add): data.zero()

		model_nd = model.getNdArray()
		data_nd = data.getNdArray()

		#data_nd[:] += np.fft.irfft(model_nd, axis=0 ,n=self.nt)*2*math.sqrt(self.nw)
		#data_nd[:] += np.fft.irfft(model_nd, axis=self.axis ,n=self.nt,norm="ortho")
		data_nd[:] += np.fft.irfft(model_nd, axis=self.axis ,n=self.nt)

		return

	# p(z,x,t) -> p(z,x,w)
	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add): model.zero()

		model_nd = model.getNdArray()
		data_nd = data.getNdArray()
		#model_nd[:] += np.fft.rfft(data_nd, axis=0)/math.sqrt(self.nt)
		#model_nd[:] += np.fft.rfft(data_nd, axis=self.axis,norm="ortho")
		model_nd[:] += np.fft.rfft(data_nd, axis=self.axis)

		return

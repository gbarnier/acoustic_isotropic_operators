import pyGF
import pyOperator as Op

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np


class gf(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,zCoordSou,xCoordSou,tstart,vel):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyGF.GF(domain,range,zCoordSou.getCpp(),xCoordSou.getCpp(),tstart,vel)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyGF.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyGF.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyGF.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

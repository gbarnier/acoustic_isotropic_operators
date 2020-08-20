# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import pySmooth2d
import pyOperator as Op

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np


class smooth2d(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,nfilt1,nfilt2):
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pySmooth2d.Smooth2d(domain,range,nfilt1,nfilt2)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySmooth2d.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySmooth2d.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pySmooth2d.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result




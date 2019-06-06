# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import pyAcoustic_iso_float_we
import pyOperator as Op

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np


class waveEquationAcousticCpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,elasticParam,n1min,n1max,n2min,n2max,n3min,n3max,boundaryCond):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(elasticParam)):
			elasticParam = elasticParam.getCpp()
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyElastic_iso_float_we.waveEquationElasticGpu(domain,range,elasticParam,n1min,n1max,n2min,n2max,n3min,n3max,boundaryCond)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_float_we.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyElastic_iso_float_we.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyElastic_iso_float_we.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

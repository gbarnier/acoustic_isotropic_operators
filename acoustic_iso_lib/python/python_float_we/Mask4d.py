# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import pyMask4d
import pyCausalMask
import pyOperator as Op

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np


class mask4d(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for 4d mask"""

	def __init__(self,domain,range,n1min,n1max,n2min,n2max,n3min,n3max,n4min,n4max,maskType):
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyMask4d.Mask4d(domain,range,n1min,n1max,n2min,n2max,n3min,n3max,n4min,n4max,maskType)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyMask4d.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyMask4d.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyMask4d.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result


class mask4d_complex(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for 4d mask"""

	def __init__(self,domain,range,n1min,n1max,n2min,n2max,n3min,n3max,n4min,n4max,maskType):
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyMask4d.Mask4d_complex(domain,range,n1min,n1max,n2min,n2max,n3min,n3max,n4min,n4max,maskType)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyMask4d.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyMask4d.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyMask4d.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

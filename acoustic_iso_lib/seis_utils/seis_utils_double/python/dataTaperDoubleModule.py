#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pydataTaperDouble
import genericIO
import SepVector
import Hypercube
import numpy as np

class dataTaperDouble(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for data tapering/muting in shot gathers"""

	def __init__(self,domain,range,maxOffset,exp,taperWidth,dataHyper,muteType):
		# Domain = Seismic data
		# Range = Seismic data
		self.setDomainRange(domain,range)
		# Checking if getCpp function is present
		if("getCpp" in dir(dataHyper)):
			dataHyper = dataHyper.getCpp()
		self.pyOp = pydataTaperDouble.dataTaperDouble(maxOffset,exp,taperWidth,dataHyper,muteType)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pydataTaperDouble.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pydataTaperDouble.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def getTaperMask(self):
		# Checking if getCpp is present
		with pydataTaperDouble.ostream_redirect():
			taperMask = self.pyOp.getTaperMask()
		return taperMask

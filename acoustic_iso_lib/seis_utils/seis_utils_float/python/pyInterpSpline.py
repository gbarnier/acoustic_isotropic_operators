#Python module encapsulating PYBIND11 module
import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np

class interpSpline1d(Op.Operator):
	"""
	   Wrapper encapsulating PYBIND11 module for cubic spline interpolation
	"""

	def __init__(self,domain):

		self.pyOp = pyInterpSpline.interpSplineLinear1d()

		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pydataTaperFloat.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pydataTaperFloat.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def getTaperMask(self):
		taperMask = genericIO.SepVector.floatVector(fromCpp=self.pyOp.getTaperMask())
		return taperMask

#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyTimeInteg
import genericIO
import SepVector
import Hypercube
import numpy as np

# Derivative in the z-direction
def timeIntegInit(args):

	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	dts=parObject.getFloat("dts")
	return dts

class timeInteg(Op.Operator):

	def __init__(self,domain,dts):

		self.setDomainRange(domain,domain)
		self.pyOp = pyTimeInteg.timeInteg(dts)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyTimeInteg.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyTimeInteg.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

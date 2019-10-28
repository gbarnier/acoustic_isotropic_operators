#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyDsoInvGpu
import genericIO
import SepVector
import Hypercube
import numpy as np

def dsoInvGpuInit(args):

	# IO object
	parObject=genericIO.io(params=args)

	nz=parObject.getInt("nz")
	nx=parObject.getInt("nx")
	nExt=parObject.getInt("nExt")
	fat=parObject.getInt("fat")
	zeroShift=parObject.getFloat("dsoZeroShift")
	return nz,nx,nExt,fat,zeroShift

class dsoInvGpu(Op.Operator):

	def __init__(self,domain,range,nz,nx,nExt,fat,zeroShift):

		self.setDomainRange(domain,range)
		self.pyOp = pyDsoInvGpu.dsoInvGpu(nz,nx,nExt,fat,zeroShift)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyDsoInvGpu.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyDsoInvGpu.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

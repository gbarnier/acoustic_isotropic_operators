#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyDsoGpu
import genericIO
import SepVector
import Hypercube
import numpy as np

def dsoGpuInit(args):

	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	nz=parObject.getInt("nz",0)
	nx=parObject.getInt("nx",0)
	nExt=parObject.getInt("nExt",0)
	fat=parObject.getInt("fat",5)
	zeroShift=parObject.getFloat("zeroShift",0.0)
	return nz,nx,nExt,fat,zeroShift

class dsoGpu(Op.Operator):

	def __init__(self,domain,range,nz,nx,nExt,fat,zeroShift):

		self.setDomainRange(domain,range)
		self.pyOp = pyDsoGpu.dsoGpu(nz,nx,nExt,fat,zeroShift)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyDsoGpu.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyDsoGpu.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

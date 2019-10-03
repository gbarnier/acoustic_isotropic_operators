#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pySpatialDeriv
import genericIO
import SepVector
import Hypercube
import numpy as np

# Derivative in the z-direction
def zGradInit(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)

	fat=parObject.getInt("fat",5)
	return fat

class zGradPython(Op.Operator):

	def __init__(self,domain,range,fat):

		self.setDomainRange(domain,range)
		self.pyOp = pySpatialDeriv.zGrad(fat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpatialDeriv.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpatialDeriv.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

# Derivative in the x-direction
def xGradInit(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)

	fat=parObject.getInt("fat",5)
	return fat

class xGradPython(Op.Operator):

	def __init__(self,domain,range,fat):

		self.setDomainRange(domain,range)
		self.pyOp = pySpatialDeriv.xGrad(fat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpatialDeriv.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpatialDeriv.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

# Derivative in the zx-direction
def zxGradInit(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)

	fat=parObject.getInt("fat",5)
	return fat

class zxGradPython(Op.Operator):

	def __init__(self,domain,range,fat):

		self.setDomainRange(domain,range)
		self.pyOp = pySpatialDeriv.zxGrad(fat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpatialDeriv.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpatialDeriv.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

# Laplacian operator
def LaplacianInit(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)

	fat=parObject.getInt("fat",5)
	return fat

class LaplacianPython(Op.Operator):

	def __init__(self,domain,range,fat):

		self.setDomainRange(domain,range)
		self.pyOp = pySpatialDeriv.Laplacian(fat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpatialDeriv.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpatialDeriv.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

# Derivative in the z-direction for extended images (for Symes' preconditioning method)
def SymesZGradInit(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)
	
	fat=parObject.getInt("fat",5)
	return fat

class SymesZGradPython(Op.Operator):

	def __init__(self,domain,fat):

		self.setDomainRange(domain,domain)
		self.pyOp = pySpatialDeriv.SymesZGrad(fat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpatialDeriv.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

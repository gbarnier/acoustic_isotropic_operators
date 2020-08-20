#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pyPadTruncateSource
import pyOperator as Op
#Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

# Used for creating a wavefield (z,x,t) from a source function (t,s) or functions
class pad_truncate_source(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,domain,range,gridPointUniqueIndex):
		#Checking if getCpp is present
		self.setDomainRange(domain,range)
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyPadTruncateSource.padTruncateSource(domain,range,gridPointUniqueIndex)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyPadTruncateSource.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyPadTruncateSource.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyPadTruncateSource.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

#Used for creating a wavefield (z,x,t,s) from a source function (t,s) or functions
class pad_truncate_source_multi_exp(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,domain,range,gridPointUniqueIndex,indexMaps):
		#Checking if getCpp is present
		self.setDomainRange(domain,range)
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyPadTruncateSource.padTruncateSource_mutli_exp(domain,range,gridPointUniqueIndex,indexMaps)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyPadTruncateSource.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyPadTruncateSource.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyPadTruncateSource.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

class pad_truncate_source_multi_exp_complex(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,domain,range,gridPointUniqueIndex,indexMaps):
		#Checking if getCpp is present
		self.setDomainRange(domain,range)
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyPadTruncateSource.padTruncateSource_mutli_exp_complex(domain,range,gridPointUniqueIndex,indexMaps)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyPadTruncateSource.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyPadTruncateSource.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyPadTruncateSource.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

class zero_pad_4d(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,domain,range):
		#Checking if getCpp is present
		self.setDomainRange(domain,range)
		self.n1out=domain.getHyper().getAxis(1).n
		self.n2out=domain.getHyper().getAxis(2).n
		self.n3out=domain.getHyper().getAxis(3).n
		self.n4out=domain.getHyper().getAxis(4).n
		return

	#pads
	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add): data.zero()

		model_nd = model.getNdArray()
		data_nd = data.getNdArray()

		data_nd[:self.n4out,:self.n3out,:self.n2out,:self.n1out] += model_nd[:]

		return

	#truncates
	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add): model.zero()

		model_nd = model.getNdArray()
		data_nd = data.getNdArray()


		model_nd[:] += data_nd[:self.n4out,:self.n3out,:self.n2out,:self.n1out]

		return

class zero_pad_2d(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,domain,range):
		#Checking if getCpp is present
		self.setDomainRange(domain,range)
		self.n1out=domain.getHyper().getAxis(1).n
		self.n2out=domain.getHyper().getAxis(2).n
		return

	#pads
	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add): data.zero()

		model_nd = model.getNdArray()
		data_nd = data.getNdArray()

		data_nd[:self.n2out,:self.n1out] += model_nd[:]

		return

	#truncates
	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add): model.zero()

		model_nd = model.getNdArray()
		data_nd = data.getNdArray()


		model_nd[:] += data_nd[:self.n2out,:self.n1out]

		return

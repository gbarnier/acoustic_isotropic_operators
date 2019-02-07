#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pydataTaperFloat
import genericIO
import SepVector
import Hypercube
import numpy as np

class dataTaperFloat(Op.Operator):
	"""
	   Wrapper encapsulating PYBIND11 module for data tapering/muting in shot gathers
	   Constructor 1 [time muting]: domain, range, t0 [s], velMute [km/s], exp [-], taperWidth [s]
	"""

	def __init__(self,*args):
		domain = args[0]
		range = args[1]
		self.setDomainRange(domain,range)

		# Constructor for time muting
		if(len(args) == 8):
			t0=args[2]
			velMute=args[3]
			exp=args[4]
			taperWidth=args[5]
			dataHyper=args[6]
			if("getCpp" in dir(dataHyper)):
				dataHyper = dataHyper.getCpp()
			moveout=args[7]
			self.pyOp = pydataTaperFloat.dataTaperFloat(t0,velMute,exp,taperWidth,dataHyper,moveout)

		# Constructor for offset muting
		elif(len(args) == 6):
			maxOffset=args[2]
			exp=args[3]
			taperWidth=args[4]
			dataHyper=args[5]
			if("getCpp" in dir(dataHyper)):
				dataHyper = dataHyper.getCpp()
			self.pyOp = pydataTaperFloat.dataTaperFloat(maxOffset,exp,taperWidth,dataHyper)

		# Wrong number of argument - Throw an error message
		else:
			raise TypeError("ERROR! Incorrect number of arguments!")

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

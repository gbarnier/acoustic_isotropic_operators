#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyDataTaper
import genericIO
import SepVector
import Hypercube
import numpy as np

def dataTaperInit(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	offset=parObject.getInt("offset",0)
	time=parObject.getInt("time",0)
	t0=parObject.getFloat("t0",0)
	velMute=parObject.getFloat("velMute",0)
	expTime=parObject.getFloat("expTime",2)
	taperWidthTime=parObject.getFloat("taperWidthTime",0)
	moveout=parObject.getString("moveout","linear")
	reverseTime=parObject.getInt("reverseTime",0)
	maxOffset=parObject.getFloat("maxOffset",0)
	expOffset=parObject.getFloat("expOffset",2)
	taperWidthOffset=parObject.getFloat("taperWidthOffset",0)
	reverseOffset=parObject.getInt("reverseOffset",0)

	return t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset

class datTaper(Op.Operator):

	def __init__(self,*args):
		domain = args[0]
		range = args[1]
		self.setDomainRange(domain,range)

		# Constructor for time and offset muting
		t0=args[2]
		velMute=args[3]
		expTime=args[4]
		taperWidthTime=args[5]
		moveout=args[6]
		reverseTime=args[7]
		maxOffset=args[8]
		expOffset=args[9]
		taperWidthOffset=args[10]
		reverseOffset=args[11]
		dataHyper=args[12]
		if("getCpp" in dir(dataHyper)):
			dataHyper = dataHyper.getCpp()
		time=args[13]
		offset=args[14]
		if (time==1 and offset==1):
			self.pyOp = pyDataTaper.dataTaper(t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,dataHyper)
		if (time==1 and offset==0):
			self.pyOp = pyDataTaper.dataTaper(t0,velMute,expTime,taperWidthTime,dataHyper,moveout,reverseTime)
		if (time==0 and offset==1):
			self.pyOp = pyDataTaper.dataTaper(maxOffset,expOffset,taperWidthOffset,dataHyper,reverseOffset)
		if (time==0 and offset==0):
			self.pyOp = pyDataTaper.dataTaper(dataHyper)

		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyDataTaper.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyDataTaper.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def getTaperMask(self):
		with pyDataTaper.ostream_redirect():
			taperMask = self.pyOp.getTaperMask()
			taperMask = SepVector.floatVector(fromCpp=taperMask)
		return taperMask

	def getTaperMaskTime(self):
		with pyDataTaper.ostream_redirect():
			taperMaskTime = self.pyOp.getTaperMaskTime()
			taperMaskTime = SepVector.floatVector(fromCpp=taperMaskTime)
		return taperMaskTime

	def getTaperMaskOffset(self):
		with pyDataTaper.ostream_redirect():
			taperMaskOffset = self.pyOp.getTaperMaskOffset()
			taperMaskOffset = SepVector.floatVector(fromCpp=taperMaskOffset)
		return taperMaskOffset


	# Wrong number of argument - Throw an error message
	# else:
	# raise TypeError("ERROR! Incorrect number of arguments!")

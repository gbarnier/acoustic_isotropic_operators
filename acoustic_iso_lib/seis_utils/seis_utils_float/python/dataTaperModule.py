#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyDataTaper
import ShotRecTaperModule
import genericIO
import SepVector
import Hypercube
import numpy as np

def dataTaperInit(args):

	# IO object
	parObject=genericIO.io(params=args)

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
	taperEndTraceWidth=parObject.getFloat("taperEndTraceWidth",0)
	streamers=parObject.getInt("streamers",0)

	#Adding shot and receiver tapering
	shotRecTaper=parObject.getInt("shotRecTaper",0)
	taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec=ShotRecTaperModule.ShotRecTaperInit(args)

	return t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,time,offset,shotRecTaper,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec,taperEndTraceWidth,streamers

class datTaper(Op.Operator):

	def __str__(self):
		"""Name of the operator"""
		return " dataTap "

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
		taperEndTraceWidth=args[22]
		streamers=args[23]
		if (time==1 and offset==1):
			self.pyOp = pyDataTaper.dataTaper(t0,velMute,expTime,taperWidthTime,moveout,reverseTime,maxOffset,expOffset,taperWidthOffset,reverseOffset,dataHyper,taperEndTraceWidth,streamers)
		if (time==1 and offset==0):
			self.pyOp = pyDataTaper.dataTaper(t0,velMute,expTime,taperWidthTime,dataHyper,moveout,reverseTime,taperEndTraceWidth)
		if (time==0 and offset==1):
			self.pyOp = pyDataTaper.dataTaper(maxOffset,expOffset,taperWidthOffset,dataHyper,reverseOffset,taperEndTraceWidth,streamers)
		if (time==0 and offset==0):
			self.pyOp = pyDataTaper.dataTaper(dataHyper,taperEndTraceWidth)
		#Checking if ShotRecTaper is requested and instantiating it
		shotRecTaper=args[15]
		if(shotRecTaper):
			taperShotWidth=args[16]
			taperRecWidth=args[17]
			expShot=args[18]
			expRec=args[19]
			edgeValShot=args[20]
			edgeValRec=args[21]
			self.ShotRecTaperOp=ShotRecTaperModule.ShotRecTaper(domain,taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec)
			#Creating temporary vector to apply chain of operators
			self.tmp_data_vec = domain.clone()

		return

	def forward(self,add,model,data):
		#Applying ShotRecTaper operator if present
		if("ShotRecTaperOp" in dir(self)):
			self.ShotRecTaperOp.forward(False,model,self.tmp_data_vec)
			model = self.tmp_data_vec.getCpp()
		else:
			if("getCpp" in dir(model)):
				model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		#Applying dataTaper operator
		with pyDataTaper.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		"""Self-adjoint operator"""
		self.forward(add,data,model)
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

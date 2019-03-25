#Python module to taper shots and receivers on the edges of the computational domain
import pyOperator as Op
import SepVector
import genericIO
import numpy as np

def ShotRecTaperInit(args):
	"""
	   Useful function to read necessary parameters for constructor
	"""
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	taperShotWidth=parObject.getFloat("taperShotWidth",0.0) # Extension of the shot dampening
	taperRecWidth=parObject.getFloat("taperRecWidth",0.0) # Extension of the receiver dampening
	expShot=parObject.getFloat("expShot",2.0) # Exponent of the cosine function for the shot taper
	expRec=parObject.getFloat("expRec",2.0) # Exponent of the cosine function for the receiver taper
	edgeValShot=parObject.getFloat("edgeValShot",0.1) # Minimum weight at the edges
	edgeValRec=parObject.getFloat("edgeValRec",0.1) # Minimum weight at the edges
	return taperShotWidth,taperRecWidth,expShot,expRec,edgeValShot,edgeValRec

class ShotRecTaper(Op.Operator):
	"""
	   Operator to taper shots and receivers on the edges of the computational domain for regular geometry (i.e., fixed regular shot-receiver sampling)
	"""
	def __init__(self,data,taperShotWidth=0,taperRecWidth=0,expShot=2,expRec=2,edgeValShot=0.1,edgeValRec=0.1):
		"""
		   Constructor for Shot and Receiver tapering:
		   data      		= [no default] - vector class; Data vector class to get axis and domain and range of the operator
		   taperShotWidth 	= [0.0] - float; Width of the shot dampening section
		   taperRecWidth  	= [0.0] - float; Width of the receiver dampening section
		   expShot   		= [2.0] - float; Exponent of the cosine function for the shot taper
		   expRec    		= [2.0] - float; Exponent of the cosine function for the receiver taper
		   edgeValShot 		= [0.2] - float; Minimum weight at the edges for Shots
		   edgeValRec 		= [0.2] - float; Minimum weight at the edges for Receivers
		"""
		if(taperShotWidth == taperRecWidth == 0):
			raise ValueError("ERROR! Provide a taperShotWidth and/or taperRecWidth different than zero!")
		if(not (0.0 <= edgeValRec < 1.0)):
			raise ValueError("ERROR! Minimum weight at the edges for receiver axis must be greater or equal than zero and smaller than one!")
		if(not (0.0 <= edgeValShot < 1.0)):
			raise ValueError("ERROR! Minimum weight at the edges for shot axis must be greater or equal than zero and smaller than one!")
		self.setDomainRange(data,data)
		dataHyper = data.getHyper()
		#Receiver axis
		nr = dataHyper.getAxis(2).n
		dr = dataHyper.getAxis(2).d
		#Shot axis
		ns = dataHyper.getAxis(3).n
		ds = dataHyper.getAxis(3).d
		#Checking that tapers are smaller than half the axis length
		if( (nr-1)*dr/2.0 < taperRecWidth ):
			raise ValueError("ERROR! Receiver-taper extent of %s greater than half receiver extent of %s"%(taperRecWidth,(nr-1)*dr/2.0))
		if( (ns-1)*ds/2.0 < taperShotWidth ):
			raise ValueError("ERROR! Shot-taper extent of %s greater than half shot extent of %s"%(taperShotWidth,(ns-1)*ds/2.0))
		#Computing cosine dampening function for receiver
		self.taperRecWidth = taperRecWidth
		if(self.taperRecWidth > 0.):
			self.RecTaperFunc = np.zeros((nr))
			for ii,pos in enumerate(np.linspace(0.,(nr-1)*dr,nr)):
				if(pos <= taperRecWidth):
					alpha0 = np.arcsin(np.power(edgeValRec,1.0/expRec))
					arg = (np.pi/2.0-alpha0)*pos/taperRecWidth+alpha0
					self.RecTaperFunc[ii] = np.power(np.sin(arg),expRec)
				elif(pos >= (nr-1)*dr-taperRecWidth):
					alpha0 = np.arccos(np.power(edgeValRec,1.0/expRec))
					arg = ((pos-((nr-1)*dr-taperRecWidth))/taperRecWidth)*alpha0
					self.RecTaperFunc[ii] = np.power(np.cos(arg),expRec)
				else:
					self.RecTaperFunc[ii] = 1.0
			#Expanding dimensions to allow broadcasting
			self.RecTaperFunc = np.expand_dims(np.expand_dims(self.RecTaperFunc,axis=0),axis=2)

		#Computing cosine dampening function for shot
		self.taperShotWidth = taperShotWidth
		if(self.taperShotWidth > 0.):
			self.ShotTaperFunc = np.zeros((ns))
			for ii,pos in enumerate(np.linspace(0.,(ns-1)*ds,ns)):
				if(pos <= taperShotWidth):
					alpha0 = np.arcsin(np.power(edgeValShot,1.0/expShot))
					arg = (np.pi/2.0-alpha0)*pos/taperShotWidth+alpha0
					self.ShotTaperFunc[ii] = np.power(np.sin(arg),expShot)
				elif(pos >= (ns-1)*ds-taperShotWidth):
					alpha0 = np.arccos(np.power(edgeValShot,1.0/expShot))
					arg = ((pos-((ns-1)*ds-taperShotWidth))/taperShotWidth)*alpha0
					self.ShotTaperFunc[ii] = np.power(np.cos(arg),expShot)
				else:
					self.ShotTaperFunc[ii] = 1.0
			#Expanding dimensions to allow broadcasting
			self.ShotTaperFunc = np.expand_dims(np.expand_dims(self.ShotTaperFunc,axis=1),axis=1)
		return


	def forward(self,add,model,data):
		"""
		   Applying the forward operator
		"""
		self.checkDomainRange(model,data)
		if(not add): data.zero()
		dataNp = data.getNdArray()
		modelNp = model.getNdArray()
		if(self.taperShotWidth > 0. and self.taperRecWidth > 0.):
			dataNp += modelNp*self.ShotTaperFunc*self.RecTaperFunc
		elif(self.taperShotWidth > 0. and self.taperRecWidth > 0.):
			dataNp += modelNp*self.ShotTaperFunc
		elif(self.taperShotWidth > 0. and self.taperRecWidth > 0.):
			dataNp += modelNp*self.RecTaperFunc
		return

	def adjoint(self,add,model,data):
		"""Self-adjoint operator"""
		self.forward(add,data,model)
		return

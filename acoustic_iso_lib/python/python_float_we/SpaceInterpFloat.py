#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pySpaceInterpFloat
import pyOperator as Op
#Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys

def space_interp_init_source(args):
	"""Function to correctly initialize space interp for single component wflds
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# elatic params
	slsq=parObject.getString("slsq", "noElasticParamFile")
	if (slsq != "noElasticParamFile"):
		vpParam=genericIO.defaultIO.getVector(slsq)

		# Horizontal axis
		nx=vpParam.getHyper().axes[1].n
		dx=vpParam.getHyper().axes[1].d
		ox=vpParam.getHyper().axes[1].o

		# vertical axis
		nz=vpParam.getHyper().axes[0].n
		dz=vpParam.getHyper().axes[0].d
		oz=vpParam.getHyper().axes[0].o
	else:
		# z Axis
		nz=parObject.getInt("nz",-1)
		oz=parObject.getFloat("oz",-1.0)
		dz=parObject.getFloat("dz",-1.0)

		# x axis
		nx=parObject.getInt("nx",-1)
		ox=parObject.getFloat("ox",-1.0)
		dx=parObject.getFloat("dx",-1.0)

	# Sources geometry
	nzSource=1
	ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
	dzSource=1
	nxSource=1
	oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
	dxSource=1
	spacingShots=parObject.getInt("spacingShots")
	nExp = parObject.getInt("nExp")
	sourceAxis=Hypercube.axis(n=nExp,o=0,d=1)

	##need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
	zAxis=Hypercube.axis(n=nz,o=oz,d=dz)
	xAxis=Hypercube.axis(n=nx,o=ox,d=dx)

	centerGridHyper=Hypercube.hypercube(axes=[zAxis,xAxis])

	#check which source injection interp method
	# sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	# sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)

	# sources _zCoord and _xCoord
	zCoordHyper=Hypercube.hypercube(axes=[sourceAxis])
	zCoordFloat=SepVector.getSepVector(zCoordHyper,storage="dataFloat")
	xCoordHyper=Hypercube.hypercube(axes=[sourceAxis])
	xCoordFloat=SepVector.getSepVector(xCoordHyper,storage="dataFloat")

	xCoordDMat=xCoordFloat.getNdArray()
	zCoordDMat=zCoordFloat.getNdArray()

	for ishot in range(nExp):
		#Setting z and x position of the source for the given experiment
		zCoordDMat[ishot]= oz+ozSource*dz
		xCoordDMat[ishot]= ox+oxSource*dx
		oxSource=oxSource+spacingShots # Shift source

	return zCoordFloat,xCoordFloat,centerGridHyper

def space_interp_init_rec(args):
	"""Function to correctly initialize space interp for multiple component wflds
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# elatic params
	slsq=parObject.getString("slsq", "noElasticParamFile")
	if (slsq != "noElasticParamFile"):
		vpParam=genericIO.defaultIO.getVector(slsq)

		# Horizontal axis
		nx=vpParam.getHyper().axes[1].n
		dx=vpParam.getHyper().axes[1].d
		ox=vpParam.getHyper().axes[1].o

		# vertical axis
		nz=vpParam.getHyper().axes[0].n
		dz=vpParam.getHyper().axes[0].d
		oz=vpParam.getHyper().axes[0].o
	else:
		# z Axis
		nz=parObject.getInt("nz",-1)
		oz=parObject.getFloat("oz",-1.0)
		dz=parObject.getFloat("dz",-1.0)

		# x axis
		nx=parObject.getInt("nx",-1)
		ox=parObject.getFloat("ox",-1.0)
		dx=parObject.getFloat("dx",-1.0)

	# rec geometry
	nzReceiver=parObject.getInt("nzReceiver")
	ozReceiver=parObject.getInt("ozReceiver")-1+parObject.getInt("zPadMinus",0)+parObject.getInt("fat")
	dzReceiver=parObject.getInt("dzReceiver")
	nxReceiver=parObject.getInt("nxReceiver")
	oxReceiver=parObject.getInt("oxReceiver")-1+parObject.getInt("xPadMinus",0)+parObject.getInt("fat")
	dxReceiver=parObject.getInt("dxReceiver")
	receiverAxis=Hypercube.axis(n=nxReceiver*nzReceiver,o=0,d=1)
	nRecGeom=1; # Constant receivers' geometry

	##need a hypercube for centerGrid, x shifted, z shifted, and xz shifted grid
	zAxis=Hypercube.axis(n=nz,o=oz,d=dz)
	xAxis=Hypercube.axis(n=nx,o=ox,d=dx)

	centerGridHyper=Hypercube.hypercube(axes=[zAxis,xAxis])

	#check which source injection interp method
	# sourceInterpMethod = parObject.getString("sourceInterpMethod","linear")
	# sourceInterpNumFilters = parObject.getInt("sourceInterpNumFilters",4)

	# sources _zCoord and _xCoord
	zCoordHyper=Hypercube.hypercube(axes=[receiverAxis])
	zCoordFloat=SepVector.getSepVector(zCoordHyper,storage="dataFloat")
	xCoordHyper=Hypercube.hypercube(axes=[receiverAxis])
	xCoordFloat=SepVector.getSepVector(xCoordHyper,storage="dataFloat")

	xCoordDMat=xCoordFloat.getNdArray()
	zCoordDMat=zCoordFloat.getNdArray()

	for irecz in range(nzReceiver):
		for irecx in range(nxReceiver):
			#Setting z and x position of the source for the given experiment
			zCoordDMat[irecx*nzReceiver + irecz]= oz + ozReceiver*dz + dzReceiver*dz*irecz
			xCoordDMat[irecx*nzReceiver + irecz]= ox + oxReceiver*dx + dxReceiver*dx*irecx

	return zCoordFloat,xCoordFloat,centerGridHyper

class space_interp(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,zCoord,xCoord,vpParamHypercube,nt,interpMethod,nFilt):
		#Checking if getCpp is present
		# self.setDomainRange(domain,range)
		# if("getCpp" in dir(domain)):
		# 	domain = domain.getCpp()
		# if("getCpp" in dir(range)):
		# 	range = range.getCpp()
		self.pyOp = pySpaceInterpFloat.spaceInterp(zCoord.getCpp(),xCoord.getCpp(),vpParamHypercube.getCpp(),nt,interpMethod,nFilt)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpaceInterpFloat.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySpaceInterpFloat.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pySpaceInterpFloat.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

	def getNDeviceIrreg(self):
		with pySpaceInterpFloat.ostream_redirect():
			result = self.pyOp.getNDeviceIrreg()
		return result

	def getNDeviceReg(self):
		with pySpaceInterpFloat.ostream_redirect():
			result = self.pyOp.getNDeviceReg()
		return result

	def getRegPosUniqueVector(self):
		with pySpaceInterpFloat.ostream_redirect():
			result = self.pyOp.getRegPosUniqueVector()
		return result

# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import pyAcoustic_iso_float_nl
import pyAcoustic_iso_float_born
import pyAcoustic_iso_float_born_ext
import pyAcoustic_iso_float_tomo
import pyAcoustic_iso_float_wemva
import pyOperator as Op

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np

from pyAcoustic_iso_float_nl import deviceGpu

############################ Bounds vectors ####################################
# Create bound vectors for FWI
def createBoundVectors(parObject,model):

	# Get model dimensions
	nz=parObject.getInt("nz")
	nx=parObject.getInt("nx")
	fat=parObject.getInt("fat")
	spline=parObject.getInt("spline",0)
	if (spline==1): fat=0

	# Min bound
	minBoundVectorFile=parObject.getString("minBoundVector","noMinBoundVectorFile")
	if (minBoundVectorFile=="noMinBoundVectorFile"):
		minBound=parObject.getFloat("minBound")
		minBoundVector=model.clone()
		minBoundVector.scale(0.0)
		minBoundVectorNd=minBoundVector.getNdArray()
		for ix in range(fat,nx-fat):
			for iz in range(fat,nz-fat):
				minBoundVectorNd[ix][iz]=minBound

	else:
		minBoundVector=genericIO.defaultIO.getVector(minBoundVectorFile)

	# Max bound
	maxBoundVectorFile=parObject.getString("maxBoundVector","noMaxBoundVectorFile")
	if (maxBoundVectorFile=="noMaxBoundVectorFile"):
		maxBound=parObject.getFloat("maxBound")
		maxBoundVector=model.clone()
		maxBoundVector.scale(0.0)
		maxBoundVectorNd=maxBoundVector.getNdArray()
		for ix in range(fat,nx-fat):
			for iz in range(fat,nz-fat):
				maxBoundVectorNd[ix][iz]=maxBound

	else:
		maxBoundVector=genericIO.defaultIO.getVector(maxBoundVectorFile)


	return minBoundVector,maxBoundVector

############################ Acquisition geometry ##############################
# Build sources geometry
def buildSourceGeometry(parObject,vel):

	# Horizontal axis
	dx=vel.getHyper().axes[1].n
	dx=vel.getHyper().axes[1].d
	ox=vel.getHyper().axes[1].o

	# Sources geometry
	nzSource=1
	ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
	dzSource=1
	nxSource=1
	oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
	dxSource=1
	spacingShots=parObject.getInt("spacingShots")
	sourceAxis=Hypercube.axis(n=parObject.getInt("nShot"),o=ox+oxSource*dx,d=spacingShots*dx)
	sourcesVector=[]
	for ishot in range(parObject.getInt("nShot")):
		sourcesVector.append(deviceGpu(nzSource,ozSource,dzSource,nxSource,oxSource,dxSource,vel.getCpp(),parObject.getInt("nts")))
		oxSource=oxSource+spacingShots # Shift source

	return sourcesVector,sourceAxis

# Build receivers geometry
def buildReceiversGeometry(parObject,vel):

	# Horizontal axis
	dx=vel.getHyper().axes[1].n
	dx=vel.getHyper().axes[1].d
	ox=vel.getHyper().axes[1].o

	nzReceiver=1
	ozReceiver=parObject.getInt("depthReceiver")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
	dzReceiver=1
	nxReceiver=parObject.getInt("nReceiver")
	oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
	dxReceiver=parObject.getInt("dReceiver")
	receiverAxis=Hypercube.axis(n=nxReceiver,o=ox+oxReceiver*dx,d=dxReceiver*dx)
	receiversVector=[]
	nRecGeom=1; # Constant receivers' geometry
	for iRec in range(nRecGeom):
		receiversVector.append(deviceGpu(nzReceiver,ozReceiver,dzReceiver,nxReceiver,oxReceiver,dxReceiver,vel.getCpp(),parObject.getInt("nts")))

	return receiversVector,receiverAxis

############################### Nonlinear ######################################
def nonlinearOpInitFloat(args):
	"""Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Allocate and read velocity
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		print("**** ERROR: User did not provide velocity file ****\n")
		quit()
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate model and fill with zeros
	dummyAxis=Hypercube.axis(n=1)
	modelHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis,dummyAxis])
	modelFloat=SepVector.getSepVector(modelHyper)

	# Allocate data and fill with zeros
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis])
	dataFloat=SepVector.getSepVector(dataHyper)

	# Outputs
	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,receiversVector

class nonlinearPropShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear propagator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,receiversVector):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		self.pyOp = pyAcoustic_iso_float_nl.nonlinearPropShotsGpu(velocity,paramP,sourceVector,receiversVector)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_nl.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

def nonlinearFwiOpInitFloat(args):
	"""Function to correctly initialize a nonlinear operator where the model is velocity
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Allocate and read starting model
	modelStartFile=parObject.getString("vel")
	modelStart=genericIO.defaultIO.getVector(modelStartFile)

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,modelStart)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,modelStart)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate wavelet and fill with zeros
	dummyAxis=Hypercube.axis(n=1)
	sourcesSignalHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis,dummyAxis])
	sourcesSignal=SepVector.getSepVector(sourcesSignalHyper)

	# Allocate data and fill with zeros
	dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis])
	dataFloat=SepVector.getSepVector(dataHyper)

	# Outputs
	return modelStart,dataFloat,sourcesSignal,parObject,sourcesVector,receiversVector

class nonlinearFwiPropShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear propagator where the model vector is the velocity"""

	def __init__(self,domain,range,sources,paramP,sourceVector,receiversVector):
		#Domain = velocity model
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(sources)):
			sources = sources.getCpp()
			self.sources = sources.clone()
		self.pyOp = pyAcoustic_iso_float_nl.nonlinearPropShotsGpu(domain,paramP,sourceVector,receiversVector)
		return

	def forward(self,add,model,data):
		#Setting velocity model
		self.setVel(model)
		#Checking if getCpp is present
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.forward(add,self.sources,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.setVel(vel)
		return

################################### Born #######################################
def BornOpInitFloat(args):
	"""Function to correctly initialize Born operator
	   The function will return the necessary variables for operator construction
	"""

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Velocity model
	velFile=parObject.getString("vel")
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Read sources signals
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

	# Allocate model
	modelFloat=SepVector.getSepVector(velFloat.getHyper())

	# Allocate data
	dataFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis]))

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector

class BornShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		self.pyOp = pyAcoustic_iso_float_born.BornShotsGpu(velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_born.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

# ############################# Born extended ####################################
def BornExtOpInitFloat(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Velocity model
	velFile=parObject.getString("vel", "noVelFile")
	if (velFile == "noVelFile"):
		print("**** ERROR: User did not provide vel file ****\n")
		quit()
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

	# Space axes
	zAxis=velFloat.getHyper().axes[0]
	xAxis=velFloat.getHyper().axes[1]

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Extension axis
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR: User did not provide extension type ****\n")
		quit()

	nExt=parObject.getInt("nExt", -1)
	if (nExt == -1):
		print("**** ERROR: User did not provide size of extension axis ****\n")
		quit()
	if (nExt%2 ==0):
		print("Length of extension axis must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt=parObject.getFloat("dts",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt=parObject.getFloat("dx",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	extAxis=Hypercube.axis(n=nExt,o=oExt,d=dExt) # Create extended axis

	# Read sources signals (we assume one unique point source signature for all shots)
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

	# Allocate model
	modelFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,extAxis]))

	# Allocate data
	dataFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis]))

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector

class BornExtShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Extended Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		self.pyOp = pyAcoustic_iso_float_born_ext.BornExtShotsGpu(velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

# ################################### Tomo #######################################
def tomoExtOpInitFloat(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Velocity model
	velFile=parObject.getString("vel", "noVelFile")
	if (velFile == "noVelFile"):
		print("**** ERROR: User did not provide vel file ****\n")
		quit()
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

	# Space axes
	zAxis=velFloat.getHyper().axes[0]
	xAxis=velFloat.getHyper().axes[1]

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Extension axis
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR: User did not provide extension type ****\n")
		quit()

	nExt=parObject.getInt("nExt", -1)
	if (nExt == -1):
		print("**** ERROR: User did not provide size of extension axis ****\n")
		quit()
	if (nExt%2 ==0):
		print("Length of extension axis must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt=parObject.getFloat("dts",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt=parObject.getFloat("dx",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	extAxis=Hypercube.axis(n=nExt,o=oExt,d=dExt) # Create extended axis

	# Read sources signals (we assume one unique point source signature for all shots)
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

	# Extended reflectivity
	reflectivityFile=parObject.getString("reflectivity","noReflectivityFile")
	if (reflectivityFile == "noReflectivityFile"):
		print("**** ERROR: User did not provide reflectivity file ****\n")
		quit()
	reflectivityFloat=genericIO.defaultIO.getVector(reflectivityFile,ndims=3)

	# Allocate model
	modelFloat=SepVector.getSepVector(velFloat.getHyper())

	# Allocate data
	dataFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis]))

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,reflectivityFloat

class tomoExtShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector,reflectivityExt):
		# Domain = Background perturbation
		# Range = Born data
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		if("getCpp" in dir(reflectivityExt)):
			reflectivityExt = reflectivityExt.getCpp()

		self.pyOp = pyAcoustic_iso_float_tomo.tomoExtShotsGpu(velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector,reflectivityExt)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

# ################################### Wemva ######################################
def wemvaExtOpInitFloat(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Velocity model
	velFile=parObject.getString("vel", "noVelFile")
	if (velFile == "noVelFile"):
		print("**** ERROR: User did not provide vel file ****\n")
		quit()
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

	# Space axes
	zAxis=velFloat.getHyper().axes[0]
	xAxis=velFloat.getHyper().axes[1]

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Extension axis
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR: User did not provide extension type ****\n")
		quit()

	nExt=parObject.getInt("nExt", -1)
	if (nExt == -1):
		print("**** ERROR: User did not provide size of extension axis ****\n")
		quit()
	if (nExt%2 ==0):
		print("Length of extension axis must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt=parObject.getFloat("dts",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt=parObject.getFloat("dx",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	extAxis=Hypercube.axis(n=nExt,o=oExt,d=dExt) # Create extended axis

	# Read sources signals (we assume one unique point source signature for all shots)
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

	# Receiver signals (Seismic data or "wemvaData") as a float3DReg
	wemvaDataFile=parObject.getString("wemvaData","noWemvaDataFile")
	if (wemvaDataFile == "noWemvaDataFile"):
		print("**** ERROR: User did not provide wemva seismic data file ****\n")
		quit()
	receiversSignalsFloat=genericIO.defaultIO.getVector(wemvaDataFile,ndims=3) 	# Read seismic data as a 3DReg
	receiversSignalsFloatNp=receiversSignalsFloat.getNdArray() # Get the numpy array of the total dataset

	# Initialize receivers signals vector
	receiversSignalsVector=[]

	# Copy wemva data to vector of 2DReg
	for iShot in range(sourceAxis.n):
		receiversSignalsSliceFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis])) # Create a 2DReg data slice
		receiversSignalsSliceFloatNp=receiversSignalsSliceFloat.getNdArray() # Get the numpy array of the slice
		for iReceiver in range(receiverAxis.n):
			for its in range(timeAxis.n):
				receiversSignalsSliceFloatNp[iReceiver][its]=receiversSignalsFloatNp[iShot][iReceiver][its]

		# Push back slice to vector after each shot
		receiversSignalsVector.append(receiversSignalsSliceFloat)

	# Allocate data
	dataFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,extAxis]))

	# Allocate model
	modelFloat=SepVector.getSepVector(velFloat.getHyper())

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,receiversSignalsVector

class wemvaExtShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector,receiversSignalsVector):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()

		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		for idx,receiversSignal in enumerate(receiversSignalsVector):
			if("getCpp" in dir(receiversSignal)):
				receiversSignalsVector[idx] = receiversSignal.getCpp()
		self.pyOp = pyAcoustic_iso_float_wemva.wemvaExtShotsGpu(velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector,receiversSignalsVector)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

#Python module encapsulating PYBIND11 module
#It seems necessary to allow std::cout redirection to screen
import pyAcoustic_iso_double1
import pyAcoustic_iso_double2
import pyAcoustic_iso_double3
import pyOperator as Op
#Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np

from pyAcoustic_iso_double1 import deviceGpu

def nonlinearOpInitDouble(args):
	"""Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
	# Preliminary steps
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	velFile=parObject.getString("vel")

	# Velocity (and convert to double precision)
	velFloat=genericIO.defaultIO.getVector(velFile) # Allocates memory for velocity model float
	velDouble=SepVector.getSepVector(velFloat.getHyper(), storage="dataDouble") # Allocates memory for velocity model double (filled with zeros)
	velDoubleNp=np.array(velDouble.getCpp(), copy=False) # Convert sep vector to numpy array (no mem allocation)
	velFloatNp=np.array(velFloat.getCpp(), copy=False)
	velDoubleNp[:]=velFloatNp

	# Sources geometry
	nzSource=1
	ozSource=parObject.getInt("zSource") - 1 + parObject.getInt("zPadMinus") + parObject.getInt("fat")
	dzSource=1
	nxSource=1
	oxSource=parObject.getInt("xSource") - 1 + parObject.getInt("xPadMinus") + parObject.getInt("fat")
	dxSource=1
	spacingShots=parObject.getInt("spacingShots")
	sourceAxis=Hypercube.axis(n=parObject.getInt("nShot"), o=oxSource, d=dxSource)
	sourcesVector=[]
	for ishot in range(parObject.getInt("nShot")):
		sourcesVector.append(deviceGpu(nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, velDouble.getCpp(), parObject.getInt("nts")))
		oxSource=oxSource+spacingShots # Shift source

	# Receivers geometry
	nzReceiver=1
	ozReceiver=parObject.getInt("depthReceiver") - 1 + parObject.getInt("zPadMinus") + parObject.getInt("fat")
	dzReceiver=1
	nxReceiver=parObject.getInt("nReceiver")
	oxReceiver=parObject.getInt("oReceiver") - 1 + parObject.getInt("xPadMinus") + parObject.getInt("fat")
	dxReceiver=parObject.getInt("dReceiver")
	receiverAxis=Hypercube.axis(n=nxReceiver, o=oxReceiver, d=dxReceiver)
	receiversVector=[]
	nRecGeom=1; # Constant receivers' geometry
	for iRec in range(nRecGeom):
		receiversVector.append(deviceGpu(nzReceiver, ozReceiver, dzReceiver, nxReceiver, oxReceiver, dxReceiver, velDouble.getCpp(), parObject.getInt("nts")))

	# Forward
	if (parObject.getInt("adj",0) == 0):

		# Read model (i.e., the wavelet) as a float1DReg
		modelFloat=genericIO.defaultIO.getVector(parObject.getString("model"))
		timeAxis=modelFloat.getHyper().axes[0]
		dummyAxis=Hypercube.axis(n=1)
		modelHyper=Hypercube.hypercube(axes=[timeAxis, dummyAxis, dummyAxis])
		modelDouble=SepVector.getSepVector(modelHyper, storage="dataDouble")
		modelDMat=modelDouble.getNdArray()
		modelSMat=modelFloat.getNdArray()

		# Copy model
		for its in range(timeAxis.n):
			modelDMat[0][0][its] = modelSMat[its]

		# Create data as double3DReg
		dataDouble=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis, receiverAxis, sourceAxis]), storage="dataDouble")

	# Adjoint
	else:

		# Read data as a float?DReg
		dataFloat=genericIO.defaultIO.getVector(parObject.getString("data"))

		# Create data as a double3DReg
		timeAxis=dataFloat.getHyper().axes[0]
		dummyAxis=Hypercube.axis(n=1)
		dataHyper=Hypercube.hypercube(axes=[timeAxis, receiverAxis, sourceAxis])
		dataDouble=SepVector.getSepVector(dataHyper, storage="dataDouble")
		dataDMat=dataDouble.getNdArray()
		dataSMat=dataFloat.getNdArray()

		# Case where there is only one source, one receiver
		if (sourceAxis.n == 1 and receiverAxis.n == 1):
			for its in range(timeAxis.n):
				dataDMat[0][0][its] = dataSMat[its]

		# Case where there is only one source, multiple receivers
		if (sourceAxis.n == 1 and receiverAxis.n > 1):
			for iReceiver in range(receiverAxis.n):
				for its in range(timeAxis.n):
					dataDMat[0][iReceiver][its] = dataSMat[iReceiver][its]

		# Case where there is multiple sources, only one receiver
		if (sourceAxis.n > 1 and receiverAxis.n == 1):
			for iSource in range(sourceAxis.n):
				for its in range(timeAxis.n):
					dataDMat[iSource][0][its] = dataSMat[iSource][its]

		# Case where there is multiple sources, multiple receivers
		if (sourceAxis.n > 1 and receiverAxis.n > 1):
			for iSource in range(sourceAxis.n):
				for iReceiver in range(receiverAxis.n):
					for its in range(timeAxis.n):
						dataDMat[iSource][iReceiver][its] = dataSMat[iSource][iReceiver][its]

		# Create model as a double3DReg with 2 dummy axes
		modelDouble=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis, dummyAxis, dummyAxis]), storage="dataDouble")

	# Output: model and data SepVectors
	return modelDouble, dataDouble, velDouble, parObject, sourcesVector, receiversVector

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
		self.pyOp = pyAcoustic_iso_double1.nonlinearPropShotsGpu(velocity,paramP,sourceVector,receiversVector)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double1.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double1.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_double1.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

def BornOpInitDouble(args):
	"""Function to correctly initialize Born operator
	   The function will return the necessary variables for operator construction
	"""
	# Preliminary steps
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	velFile=parObject.getString("vel")
	sourcesFile=parObject.getString("sources")

	# Velocity model
	velFloat=genericIO.defaultIO.getVector(velFile)
	velDouble=SepVector.getSepVector(velFloat.getHyper(), storage="dataDouble")
	velDoubleNp=np.array(velDouble.getCpp(), copy=False)
	velFloatNp=np.array(velFloat.getCpp(), copy=False)
	velDoubleNp[:]=velFloatNp

	# Sources signals (we assume one unique point source signature for all shots)
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile)
	timeAxis=sourcesSignalsFloat.getHyper().axes[0]
	dummyAxis=Hypercube.axis(n=1)
	sourcesSignalsHyper=Hypercube.hypercube(axes=[timeAxis, dummyAxis]) # Make the wavelet a float2DReg/double2DReg
	sourcesSignalsDouble=SepVector.getSepVector(sourcesSignalsHyper, storage="dataDouble") # Allocates memory for velocity model double (filled with zeros)
	sourcesSignalsDoubleNp=np.array(sourcesSignalsDouble.getCpp(), copy=False) # Convert sep vector to numpy array (no mem allocation)
	sourcesSignalsFloatNp=np.array(sourcesSignalsFloat.getCpp(), copy=False)
	for its in range(timeAxis.n):
		sourcesSignalsDoubleNp[0][its]=sourcesSignalsFloatNp[its]
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsDouble)

	# Sources geometry
	nzSource = 1
	ozSource =parObject.getInt("zSource") - 1 + parObject.getInt("zPadMinus") + parObject.getInt("fat")
	dzSource = 1
	nxSource = 1
	oxSource =parObject.getInt("xSource") - 1 + parObject.getInt("xPadMinus") + parObject.getInt("fat")
	dxSource = 1
	spacingShots = parObject.getInt("spacingShots")
	sourceAxis=Hypercube.axis(n=parObject.getInt("nShot"), o=oxSource, d=dxSource)
	sourcesVector=[]
	for ishot in range(parObject.getInt("nShot")):
		sourcesVector.append(deviceGpu(nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, velDouble.getCpp(), parObject.getInt("nts")))
		oxSource = oxSource + spacingShots

	# Receivers geometry
	nzReceiver = 1
	ozReceiver = parObject.getInt("depthReceiver") - 1 + parObject.getInt("zPadMinus") + parObject.getInt("fat")
	dzReceiver = 1
	nxReceiver = parObject.getInt("nReceiver")
	oxReceiver = parObject.getInt("oReceiver") - 1 + parObject.getInt("xPadMinus") + parObject.getInt("fat")
	dxReceiver = parObject.getInt("dReceiver")
	receiverAxis=Hypercube.axis(n=nxReceiver, o=oxReceiver, d=dxReceiver)
	receiversVector=[]
	nRecGeom = 1; # Constant receivers' geometry
	for iRec in range(nRecGeom):
		receiversVector.append(deviceGpu(nzReceiver, ozReceiver, dzReceiver, nxReceiver, oxReceiver, dxReceiver, velDouble.getCpp(), parObject.getInt("nts")))

	# Forward: read wavelet and allocate data (filled with zeros)
	if (parObject.getInt("adj", 0) == 0):

		# Read model (i.e., the reflectivity) as a float vector
		modelFloat=genericIO.defaultIO.getVector(parObject.getString("model"))
		modelDouble=SepVector.getSepVector(modelFloat.getHyper(), storage="dataDouble")
		modelDoubleNp=np.array(modelDouble.getCpp(), copy=False)
		modelFloatNp=np.array(modelFloat.getCpp(), copy=False)
		modelDoubleNp[:]=modelFloatNp

		# Allocate dataDouble and fill with zeros
		dataDouble=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis, receiverAxis, sourceAxis]), storage="dataDouble")

	# Adjoint: read data and allocate model (filled with zeros)
	else:

		# Read data as a float vector
		dataFloat=genericIO.defaultIO.getVector(parObject.getString("data"))

		# Create data vector in double precision
		dataHyper=Hypercube.hypercube(axes=[timeAxis, receiverAxis, sourceAxis])
		dataDouble=SepVector.getSepVector(dataHyper, storage="dataDouble")
		dataDMat=dataDouble.getNdArray()
		dataSMat=dataFloat.getNdArray()

		# Case where there is only one source, one receiver
		if (sourceAxis.n == 1 and receiverAxis.n == 1):
			for its in range(timeAxis.n):
				dataDMat[0][0][its] = dataSMat[its]

		# Case where there is only one source, multiple receivers
		if (sourceAxis.n == 1 and receiverAxis.n > 1):
			for iReceiver in range(receiverAxis.n):
				for its in range(timeAxis.n):
					dataDMat[0][iReceiver][its] = dataSMat[iReceiver][its]

		# Case where there is only multiple sources, one receiver
		if (sourceAxis.n > 1 and receiverAxis.n == 1):
			for iSource in range(sourceAxis.n):
				for its in range(timeAxis.n):
					dataDMat[iSource][0][its] = dataSMat[iSource][its]

		# Case where there is multiple sources, multiple receivers
		if (sourceAxis.n > 1 and receiverAxis.n > 1):
			for iSource in range(sourceAxis.n):
				for iReceiver in range(receiverAxis.n):
					for its in range(timeAxis.n):
						dataDMat[iSource][iReceiver][its] = dataSMat[iSource][iReceiver][its]


		# Allocate modelDouble and fill with zeros
		modelDouble=SepVector.getSepVector(velDouble.getHyper(), storage="dataDouble")

	return modelDouble, dataDouble, velDouble, parObject, sourcesVector, sourcesSignalsVector, receiversVector

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
		self.pyOp = pyAcoustic_iso_double2.BornShotsGpu(velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double2.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double2.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_double2.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

def BornExtOpInitDouble(args):

	# Preliminary steps
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()
	velFile=parObject.getString("vel")
	sourcesFile=parObject.getString("sources")

	# Velocity model
	velFloat=genericIO.defaultIO.getVector(velFile)
	velDouble=SepVector.getSepVector(velFloat.getHyper(), storage="dataDouble")
	velDoubleNp=np.array(velDouble.getCpp(), copy=False)
	velFloatNp=np.array(velFloat.getCpp(), copy=False)
	velDoubleNp[:]=velFloatNp
	zAxis=velDouble.getHyper().axes[0]
	xAxis=velDouble.getHyper().axes[1]

	# Extension axis
	extension=parObject.getString("extension")
	nExt=parObject.getInt("nExt")
	if (nExt%2 ==0):
		print("Length of extension axis must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt=parObject.getInt("dts")
		nExt=parObject.getInt("nExt")
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt=parObject.getInt("dx")
		nExt=parObject.getInt("nExt")
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	extAxis=Hypercube.axis(n=nExt, o=oExt, d=dExt) # Create extended axis

	# Sources signals (we assume one unique point source signature for all shots)
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile)
	timeAxis=sourcesSignalsFloat.getHyper().axes[0]
	dummyAxis=Hypercube.axis(n=1)
	sourcesSignalsHyper=Hypercube.hypercube(axes=[timeAxis, dummyAxis])
	sourcesSignalsDouble=SepVector.getSepVector(sourcesSignalsHyper, storage="dataDouble")
	sourcesSignalsDoubleNp=np.array(sourcesSignalsDouble.getCpp(), copy=False)
	sourcesSignalsFloatNp=np.array(sourcesSignalsFloat.getCpp(), copy=False)
	for its in range(timeAxis.n):
		sourcesSignalsDoubleNp[0][its]=sourcesSignalsFloatNp[its]
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsDouble)

	# Sources geometry
	nzSource=1
	ozSource=parObject.getInt("zSource") - 1 + parObject.getInt("zPadMinus") + parObject.getInt("fat")
	dzSource=1
	nxSource=1
	oxSource=parObject.getInt("xSource") - 1 + parObject.getInt("xPadMinus") + parObject.getInt("fat")
	dxSource=1
	spacingShots=parObject.getInt("spacingShots")
	sourceAxis=Hypercube.axis(n=parObject.getInt("nShot"), o=oxSource, d=dxSource)
	sourcesVector=[]
	for ishot in range(parObject.getInt("nShot")):
		sourcesVector.append(deviceGpu(nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, velDouble.getCpp(), parObject.getInt("nts")))
		oxSource=oxSource + spacingShots

	# Receivers geometry
	nzReceiver = 1
	ozReceiver = parObject.getInt("depthReceiver") - 1 + parObject.getInt("zPadMinus") + parObject.getInt("fat")
	dzReceiver = 1
	nxReceiver = parObject.getInt("nReceiver")
	oxReceiver = parObject.getInt("oReceiver") - 1 + parObject.getInt("xPadMinus") + parObject.getInt("fat")
	dxReceiver = parObject.getInt("dReceiver")
	receiverAxis=Hypercube.axis(n=nxReceiver, o=oxReceiver, d=dxReceiver)
	receiversVector=[]
	nRecGeom = 1; # Constant receivers' geometry
	for iRec in range(nRecGeom):
		receiversVector.append(deviceGpu(nzReceiver, ozReceiver, dzReceiver, nxReceiver, oxReceiver, dxReceiver, velDouble.getCpp(), parObject.getInt("nts")))

	# Forward: read wavelet and allocate data (filled with zeros)
	if (parObject.getInt("adj", 0) == 0):

		# Read model (i.e., the exended reflectivity) as a float vector
		modelFloat=genericIO.defaultIO.getVector(parObject.getString("model"))
		if (modelFloat.getHyper().getNdim() == 2):
			modelDouble=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis, xAxis, extAxis]), storage="dataDouble")
		else:
			modelDouble=SepVector.getSepVector(modelFloat.getHyper(), storage="dataDouble")

		modelDoubleNp=modelDouble.getNdArray()
		modelFloatNp=modelFloat.getNdArray()
		modelDoubleNp[:]=modelFloatNp

		# Allocate dataDouble and fill with zeros
		dataDouble=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis, receiverAxis, sourceAxis]), storage="dataDouble")

	# Adjoint: read data and allocate model (filled with zeros)
	else:

		# Read data as a float vector
		dataFloat=genericIO.defaultIO.getVector(parObject.getString("data"))

		# Create data vector in double precision
		dataHyper=Hypercube.hypercube(axes=[timeAxis, receiverAxis, sourceAxis])
		dataDouble=SepVector.getSepVector(dataHyper, storage="dataDouble")
		dataDMat=dataDouble.getNdArray()
		dataSMat=dataFloat.getNdArray()

		# Case where there is only one source, one receiver
		if (sourceAxis.n == 1 and receiverAxis.n == 1):
			for its in range(timeAxis.n):
				dataDMat[0][0][its] = dataSMat[its]

		# Case where there is only one source, multiple receivers
		if (sourceAxis.n == 1 and receiverAxis.n > 1):
			for iReceiver in range(receiverAxis.n):
				for its in range(timeAxis.n):
					dataDMat[0][iReceiver][its] = dataSMat[iReceiver][its]

		# Case where there is only multiple sources, one receiver
		if (sourceAxis.n > 1 and receiverAxis.n == 1):
			for iSource in range(sourceAxis.n):
				for its in range(timeAxis.n):
					dataDMat[iSource][0][its] = dataSMat[iSource][its]

		# Case where there is multiple sources, multiple receivers
		if (sourceAxis.n > 1 and receiverAxis.n > 1):
			for iSource in range(sourceAxis.n):
				for iReceiver in range(receiverAxis.n):
					for its in range(timeAxis.n):
						dataDMat[iSource][iReceiver][its] = dataSMat[iSource][iReceiver][its]


		# Allocate modelDouble and fill with zeros
		modelHyper=Hypercube.hypercube(axes=[zAxis, xAxis, extAxis])
		modelDouble=SepVector.getSepVector(modelHyper, storage="dataDouble")

	return modelDouble, dataDouble, velDouble, parObject, sourcesVector, sourcesSignalsVector, receiversVector

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
		self.pyOp = pyAcoustic_iso_double3.BornExtShotsGpu(velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double3.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_double3.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_double3.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

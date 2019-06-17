# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import pyAcoustic_iso_float_we
import pyOperator as Op

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np

def waveEquationOpInitFloat(args):
	"""Function to correctly initialize wave equation operator
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# elatic params
	slsq=parObject.getString("slsq", "noElasticParamFile")
	if (slsq == "noElasticParamFile"):
		print("**** ERROR: User did not provide elastic parameter file ****\n")
		sys.exit()
	slsqFloat=genericIO.defaultIO.getVector(slsq)
	# elasticParamDouble=SepVector.getSepVector(elasticParamFloat.getHyper(),storage="dataDouble")
	# elasticParamDoubleNp=elasticParamDouble.getNdArray()
	# elasticParamFloatNp=elasticParamFloat.getNdArray()
	# elasticParamDoubleNp[:]=elasticParamFloatNp

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# z Axis
	nz=parObject.getInt("nz",-1)
	oz=parObject.getFloat("oz",-1.0)
	dz=parObject.getFloat("dz",-1.0)
	zAxis=Hypercube.axis(n=nz,o=oz,d=dz)

	# x axis
	nx=parObject.getInt("nx",-1)
	ox=parObject.getFloat("ox",-1.0)
	dx=parObject.getFloat("dx",-1.0)
	xAxis=Hypercube.axis(n=nx,o=ox,d=dx)

	#wavefield axis
	wavefieldAxis=Hypercube.axis(n=5)

	# Allocate model
	modelHyper=Hypercube.hypercube(axes=[timeAxis,zAxis,xAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[timeAxis,zAxis,xAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# Initialize operator
	boundCond=parObject.getInt("boundCond",0)
	if(boundCond==0):
		n1min=1
		n1max=dataHyper.axes[0].n-2
		n2min=6
		n2max=dataHyper.axes[1].n-7
		n3min=6
		n3max=dataHyper.axes[2].n-7
	else:
		padWfld=parObject.getInt("padWfld",0)
		n1min=0
		n1max=dataHyper.axes[0].n-2
		n2min=padWfld
		n2max=dataHyper.axes[1].n-padWfld-1
		n3min=padWfld
		n3max=dataHyper.axes[2].n-padWfld-1
	op=waveEquationAcousticCpu(modelFloat,dataFloat,slsqFloat,n1min,n1max,n2min,n2max,n3min,n3max,boundCond)

	# Outputs
	return modelFloat,dataFloat,slsqFloat,parObject,op

class waveEquationAcousticCpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,slsqFloat,n1min,n1max,n2min,n2max,n3min,n3max,boundaryCond):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(slsqFloat)):
			slsqFloat = slsqFloat.getCpp()
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyAcoustic_iso_float_we.WaveReconV2(domain,range,slsqFloat,n1min,n1max,n2min,n2max,n3min,n3max,boundaryCond)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_we.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_we.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_we.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

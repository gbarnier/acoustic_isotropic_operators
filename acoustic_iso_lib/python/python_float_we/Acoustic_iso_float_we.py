# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import pyAcoustic_iso_float_we
import pyOperator as Op

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np

class windowData(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,domain,f1,f2,f3,j1,j2,j3):
		#Checking if getCpp is present
		domainNd=domain.getNdArray()
		axis0=Hypercube.axis(n=domain.getHyper().axes[0].n/j1,o=f1*domain.getHyper().axes[0].d-domain.getHyper().axes[0].o,d=j1*domain.getHyper().axes[0].d)
		axis1=Hypercube.axis(n=domain.getHyper().axes[1].n/j2,o=f2*domain.getHyper().axes[1].d-domain.getHyper().axes[1].o,d=j2*domain.getHyper().axes[1].d)
		axis2=Hypercube.axis(n=domain.getHyper().axes[2].n/j3,o=f3*domain.getHyper().axes[2].d-domain.getHyper().axes[2].o,d=j3*domain.getHyper().axes[2].d)
		rangeHyper=Hypercube.hypercube(axes=[axis0,axis1,axis2])
		range=SepVector.getSepVector(rangeHyper,storage="dataFloat")

		self.setDomainRange(domain,range)
		self.f1=f1
		self.f2=f2
		self.f3=f3
		self.j1=j1
		self.j2=j2
		self.j3=j3
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add):
			data.zero()
		data.getNdArray()[:] = model.getNdArray()[f1::j1,f2::j2,f3::j3]

		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add):
			model.zero()
		model.getNdArray()[f1::j1,f2::j2,f3::j3] = data.getNdArray()[:]
		return
class Mask3d(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module"""

	def __init__(self,domain,min1,max1,min2,j1,j2,j3):
		#Checking if getCpp is present
		domainNd=domain.getNdArray()
		axis0=Hypercube.axis(n=domain.getHyper().axes[0].n/j1,o=f1*domain.getHyper().axes[0].d-domain.getHyper().axes[0].o,d=j1*domain.getHyper().axes[0].d)
		axis1=Hypercube.axis(n=domain.getHyper().axes[1].n/j2,o=f2*domain.getHyper().axes[1].d-domain.getHyper().axes[1].o,d=j2*domain.getHyper().axes[1].d)
		axis2=Hypercube.axis(n=domain.getHyper().axes[2].n/j3,o=f3*domain.getHyper().axes[2].d-domain.getHyper().axes[2].o,d=j3*domain.getHyper().axes[2].d)
		rangeHyper=Hypercube.hypercube(axes=[axis0,axis1,axis2])
		range=SepVector.getSepVector(rangeHyper,storage="dataFloat")

		self.setDomainRange(domain,range)
		self.f1=f1
		self.f2=f2
		self.f3=f3
		self.j1=j1
		self.j2=j2
		self.j3=j3
		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add):
			data.zero()
		data.getNdArray()[:] = model.getNdArray()[f1::j1,f2::j2,f3::j3]

		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if(not add):
			model.zero()
		model.getNdArray()[f1::j1,f2::j2,f3::j3] = data.getNdArray()[:]
		return

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
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
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
	elif(boundCond==1):
		n1min=0
		n1max=dataHyper.axes[0].n
		n2min=0
		n2max=dataHyper.axes[1].n
		n3min=0
		n3max=dataHyper.axes[2].n
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
		self.pyOp = pyAcoustic_iso_float_we.WaveReconV3(domain,range,slsqFloat,n1min,n1max,n2min,n2max,n3min,n3max,boundaryCond)
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

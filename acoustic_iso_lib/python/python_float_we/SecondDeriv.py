# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import pySecondDeriv_V2
import pySecondDeriv_multi_exp_V2
import pySecondDeriv_multi_exp_freq
import pyOperator as Op

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import sys
import fft_wfld


class second_deriv(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range):
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pySecondDeriv_V2.SecondDeriv_V2(domain,range)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySecondDeriv_V2.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySecondDeriv_V2.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pySecondDeriv_V2.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

class second_deriv_multi_exp(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range):
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pySecondDeriv_multi_exp_V2.SecondDeriv_multi_exp_V2(domain,range)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySecondDeriv_multi_exp_V2.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySecondDeriv_multi_exp_V2.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pySecondDeriv_multi_exp_V2.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

class second_deriv_multi_exp_freq(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range):
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pySecondDeriv_multi_exp_freq.SecondDeriv_multi_exp_freq(domain,range)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySecondDeriv_multi_exp_freq.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pySecondDeriv_multi_exp_freq.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pySecondDeriv_multi_exp_freq.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

def secondDerivOpInitFloat_multi_exp_freq(args):
	"""Function to correctly initialize wave equation operator that runs multiple experiments in parallel
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Experiment Axis
	nExp=parObject.getInt("nExp",-1)
	expAxis=Hypercube.axis(n=nExp,o=0,d=1)

	# Time and freq Axes
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	#odd
	nf=nts//2+1 # number of samples in Fourier domain
	fs=1/dts #sampling rate of time domain
	if(nts%2 == 0): #even input
		f_range = fs/2
	else: # odd input
		f_range = fs/2*(nts-1)/nts
	df = f_range/nf
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	wAxis=Hypercube.axis(n=nf,o=0,d=df)
	print('nts:'+str(nts)+' ots:'+str(ots)+' dts:' + str(dts))
	print('nf:'+str(nf)+'f_range: '+ str(f_range) + ' of:0 df:' + str(df))

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

	# Allocate model
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis,wAxis,expAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataComplex")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,wAxis,expAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataComplex")

	# Allocate time model
	timeHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis,expAxis])
	timeFloat=SepVector.getSepVector(timeHyper,storage="dataFloat")


	# init fft op
	fftOp = fft_wfld.fft_wfld(modelFloat,timeFloat,axis=1)

	secondDerivMultExpOp = second_deriv_multi_exp_freq(modelFloat,dataFloat)

	# Outputs
	return modelFloat,dataFloat,parObject,secondDerivMultExpOp,fftOp,timeFloat

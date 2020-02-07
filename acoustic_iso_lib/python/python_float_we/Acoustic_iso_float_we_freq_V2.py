# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import sys
import pyAcoustic_iso_float_we_freq
import pyOperator as Op
import PadTruncateSource
import time

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import fft_wfld
import numpy as np

def waveEquationOpInitFloat_multi_exp_freq_V2(args):
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

	freq_np_axis = np.fft.rfftfreq(nts,dts)
	df = freq_np_axis[1]-freq_np_axis[0]
	nf= freq_np_axis.size
	# print('np axis - nf:'+str(nf)+' df:'+str(df))
	# nf=nts//2+1 # number of samples in Fourier domain
	# fs=1/dts #sampling rate of time domain
	# if(nts%2 == 0): #even input
	# 	f_range = fs/2
	# else: # odd input
	# 	f_range = fs/2*(nts-1)/nts
	# df = f_range/nf
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)
	wAxis=Hypercube.axis(n=nf,o=0,d=df)



	fmax=parObject.getFloat("fmax",-1.0)
	if(fmax!=-1.0):
		nf_wind=min(int(fmax/df)+1,nf)
		print("\tfmax selected: ",fmax)
		print("\tnew nf selected: ",nf_wind)
		wAxis_wind=Hypercube.axis(n=nf_wind,o=0,d=df)


	# print('nts:'+str(nts)+' ots:'+str(ots)+' dts:' + str(dts))
	# print('nf:'+str(nf)+'f_range: '+ str(f_range) + ' of:0 df:' + str(df))

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

	# elatic params
	slsq=parObject.getString("slsq", "noElasticParamFile")
	if (slsq == "noElasticParamFile"):
		print("**** WARNING: User did not provide slsq file. Initializing with water vel ****")
		slsqHyper=Hypercube.hypercube(axes=[zAxis,xAxis])
		slsqFloat=SepVector.getSepVector(slsqHyper,storage="dataFloat")
	else: slsqFloat=genericIO.defaultIO.getVector(slsq)

	# init fft op
	fftOpTemp = fft_wfld.fft_wfld(modelFloat,timeFloat,axis=1)
	if(fmax!=-1.0):
		modelHyper_wind=Hypercube.hypercube(axes=[zAxis,xAxis,wAxis_wind,expAxis])
		modelFloat_wind=SepVector.getSepVector(modelHyper_wind,storage="dataComplex")
		padTruncateFreqOp=PadTruncateSource.zero_pad_4d(modelFloat_wind,modelFloat)
		fftOp=fftOpTemp*padTruncateFreqOp
		modelFloat=modelFloat_wind

		dataHyper_wind=Hypercube.hypercube(axes=[zAxis,xAxis,wAxis_wind,expAxis])
		dataFloat=SepVector.getSepVector(dataHyper_wind,storage="dataComplex")
	else:
		fftOp=fftOpTemp

	dt_of_prop=parObject.getFloat("dt_of_prop",0.0000001)

	# Initialize operator on cpu
	weOp=waveEquationAcousticCpu_multi_exp_freq_V2(modelFloat,dataFloat,slsqFloat,dt_of_prop)

	# Outputs
	return modelFloat,dataFloat,slsqFloat,parObject,weOp,fftOp,timeFloat

class waveEquationAcousticCpu_multi_exp_freq_V2(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,slsqFloat,dt_of_prop=0.0000001):
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
		self.pyOp = pyAcoustic_iso_float_we_freq.WaveRecon_freq_multi_exp_V2(domain,range,slsqFloat,dt_of_prop)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_we_freq.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_we_freq.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return
	def update_slsq(self,new_slsq):
		if("getCpp" in dir(new_slsq)):
			new_slsq = new_slsq.getCpp()
		with pyAcoustic_iso_float_we_freq.ostream_redirect():
			self.pyOp.set_slsq(new_slsq)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_we_freq.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

	def dotTest(self,verb=False,maxError=1e-4):
		"""
		   Function to perform dot-product test:
		   verb     = [False] - boolean; Flag to print information to screen as the method is being run
		   maxError = [1e-4] - float; The function throws a Warning if the relative error is greater than maxError
		"""
		if(verb): print("Dot-product test of forward and adjoint operators")
		if(verb): print("-------------------------------------------------")
		#Allocating temporary vectors for dot-product test
		d1=self.domain.clone()
		d2=self.domain.clone()
		r1=self.range.clone()
		r2=self.range.clone()

		#Randomize the input vectors
		d1.rand()
		r1.rand()
		d1_ndArray = d1.getNdArray()
		r1_ndArray = r1.getNdArray()
		d1_ndArray[:,:,0:5,:]=0
		d1_ndArray[:,:,-5:-1,:]=0
		d1_ndArray[:,:,:,0:5]=0
		d1_ndArray[:,:,:,-5:-1]=0
		r1_ndArray[:,:,0:5,:]=0
		r1_ndArray[:,:,-5:-1,:]=0
		r1_ndArray[:,:,:,0:5]=0
		r1_ndArray[:,:,:,-5:-1]=0

		#Applying forward and adjoint operators with add=False
		if(verb): print("Applying forward operator add=False")
		start = time.time()
		self.forward(False,d1,r2)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))
		if(verb): print("Applying adjoint operator add=False")
		start = time.time()
		self.adjoint(False,d2,r1)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))

		#Computing dot products
		dt1=d1.dot(d2)
		dt2=r1.dot(r2)

		#Dot-product testing
		if(verb): print("Dot products add=False: domain=%s range=%s "%(dt1,dt2))
		if(verb): print("Absolute error: %s"%(abs(dt1-dt2)))
		if(verb): print("Relative error: %s \n"%(abs((dt1-dt2)/dt2)))
		if (abs((dt1-dt2)/dt1) > maxError):
			#Deleting temporary vectors
			del d1,d2,r1,r2
			raise Warning("Dot products failure add=False; relative error greater than tolerance of %s"%(maxError))

		#Applying forward and adjoint operators with add=True
		if(verb): print("\nApplying forward operator add=True")
		start = time.time()
		self.forward(True,d1,r2)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))
		if(verb): print("Applying adjoint operator add=True")
		start = time.time()
		self.adjoint(True,d2,r1)
		end = time.time()
		if(verb): print("	Runs in: %s seconds"%(end-start))

		#Computing dot products
		dt1=d1.dot(d2)
		dt2=r1.dot(r2)

		if(verb): print("Dot products add=True: domain=%s range=%s "%(dt1,dt2))
		if(verb): print("Absolute error: %s"%(abs(dt1-dt2)))
		if(verb): print("Relative error: %s \n"%(abs((dt1-dt2)/dt2)))
		if(abs((dt1-dt2)/dt1) > maxError):
			#Deleting temporary vectors
			del d1,d2,r1,r2
			raise Warning("Dot products failure add=True; relative error greater than tolerance of %s"%(maxError))

		if(verb): print("-------------------------------------------------")

		#Deleting temporary vectors
		del d1,d2,r1,r2
		return

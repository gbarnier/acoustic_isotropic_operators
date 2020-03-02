# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import sys
import pyAcoustic_iso_float_we
import pyOperator as Op
import time
import PadTruncateSource
import fft_wfld

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

def waveEquationOpInitFloat_freq(args):
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
	weOp=waveEquationAcousticCpu_multi_exp_freq(modelFloat,dataFloat,slsqFloat,dt_of_prop)

	# Outputs
	return modelFloat,dataFloat,slsqFloat,parObject,weOp,fftOp,timeFloat

def waveEquationOpInitFloat_time(args):
	"""Function to correctly initialize wave equation operator that runs multiple experiments in parallel
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

	# Experiment Axis
	nExp=parObject.getInt("nExp",-1)
	expAxis=Hypercube.axis(n=nExp,o=0,d=1)

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

	# Allocate model
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis,expAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis,expAxis])
	dataFloat=SepVector.getSepVector(dataHyper,storage="dataFloat")

	# elatic params
	slsq=parObject.getString("slsq", "noElasticParamFile")
	if (slsq == "noElasticParamFile"):
		print("**** WARNING: User did not provide slsq file. Initializing with water vel ****\n")
		slsqHyper=Hypercube.hypercube(axes=[zAxis,xAxis])
		slsqFloat=SepVector.getSepVector(slsqHyper,storage="dataFloat")
	else: slsqFloat=genericIO.defaultIO.getVector(slsq)

	gpuEnable=parObject.getInt("gpuEnable",0)
	if(gpuEnable==1):
		print("GPU ENABLED")
		# Initialize operator on gpu
		op=waveEquationAcousticGpu(modelFloat,dataFloat,slsqFloat,parObject)
	else:
		# Initialize operator on cpu
		U0=parObject.getFloat("U0",0.001)
		alpha=parObject.getFloat("alpha",0.25)
		spongeWidth=parObject.getInt("spongeWidth",0)
		print("--- Sponge Boundary Conditions ---")
		print("U_0: ", U0)
		print("alpha: ", alpha)
		print("BoundaryWidth: ", spongeWidth)
		op=waveEquationAcoustic_time(modelFloat,dataFloat,slsqFloat,U0,alpha,spongeWidth)

	# Outputs
	return modelFloat,dataFloat,slsqFloat,parObject,op

class waveEquationAcousticCpu_multi_exp_freq(Op.Operator):
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
		self.pyOp = pyAcoustic_iso_float_we.WaveRecon_freq(domain,range,slsqFloat,dt_of_prop)
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
	def update_slsq(self,new_slsq):
		if("getCpp" in dir(new_slsq)):
			new_slsq = new_slsq.getCpp()
		with pyAcoustic_iso_float_we.ostream_redirect():
			self.pyOp.set_slsq(new_slsq)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_we.ostream_redirect():
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


class waveEquationAcoustic_time(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,slsqFloat,U0,alpha,spongeWidth):
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
		self.pyOp = pyAcoustic_iso_float_we.WaveRecon_multi_exp(domain,range,slsqFloat,U0,alpha,spongeWidth)
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
	def update_slsq(self,new_slsq):
		if("getCpp" in dir(new_slsq)):
			new_slsq = new_slsq.getCpp()
		with pyAcoustic_iso_float_we.ostream_redirect():
			self.pyOp.set_slsq(new_slsq)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_we.ostream_redirect():
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
		print(d1_ndArray.shape)
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

class waveEquationAcousticGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for elastic wave equation"""

	def __init__(self,domain,range,slsqFloat,paramP):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(slsqFloat)):
			slsqFloat = slsqFloat.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		self.pyOp = pyAcoustic_iso_float_we.waveEquationAcousticGpu(domain,range,slsqFloat,paramP)
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

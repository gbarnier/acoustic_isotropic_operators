# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import pyGradio
import pyOperator as Op
import sys

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np
import pyOperator as pyOp
import wriUtilFloat
import Laplacian2d
import Mask3d
import Mask4d
import fft_wfld
import PadTruncateSource

def gradioOpInitFloat(args):
	"""Function to correctly initialize wave equation operator
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

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
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
	dataFloatInit=SepVector.getSepVector(dataHyper,storage="dataFloat")
	dataFloatInit.scale(0.0)

	# Read In pressure data
	pressureDataFile=parObject.getString("pressureData", "noPressureData")
	if (pressureDataFile == "noPressureData"):
		print("**** WARNING: User did not provide pressureData file. Using zero wfld ****\n")
		pressureDataInit=dataFloatInit.clone()
	else:
		pressureDataInit=genericIO.defaultIO.getVector(pressureDataFile)

	# Read in wavelet and make forcing term
	fullPrior = parObject.getString("fullPrior","none")
	if(fullPrior=="none"):
		print("prior from wavelet")
		_,priorTmp= wriUtilFloat.forcing_term_op_init_m(args)
		prior=priorTmp.clone()
	else:
		print("full prior")
		priorTmp=genericIO.defaultIO.getVector(fullPrior)
		prior=priorTmp.clone()

	# calculate data
	laplPressureData = pressureDataInit.clone()
	laplOp = Laplacian2d.laplacian2d(pressureDataInit,laplPressureData)
	if (parObject.getInt("dp",0)==1):
		print("Laplacian DP test:")
		laplOp.dotTest(1)
	laplOp.forward(0,pressureDataInit,laplPressureData)
	dataFloatInit.scale(0.0)
	dataFloatInit.scaleAdd(prior,1,1)
	dataFloatInit.scaleAdd(laplPressureData,1,1)

	#init mask op
	maskWidthSpace = parObject.getInt("maskWidth",15)
	mask3dOp= Mask3d.mask3d(dataFloatInit,dataFloatInit,maskWidthSpace,dataFloatInit.getHyper().axes[0].n-maskWidthSpace,maskWidthSpace,dataFloatInit.getHyper().axes[1].n-maskWidthSpace,0,dataFloatInit.getHyper().axes[2].n,0)
	mask3dOpPrep= Mask3d.mask3d(dataFloatInit,dataFloatInit,maskWidthSpace,dataFloatInit.getHyper().axes[0].n-maskWidthSpace,maskWidthSpace,dataFloatInit.getHyper().axes[1].n-maskWidthSpace,0,dataFloatInit.getHyper().axes[2].n,0)
	#mask everything
	dataFloat = dataFloatInit.clone()
	mask3dOp.forward(0,dataFloatInit,dataFloat)
	pressureData = pressureDataInit.clone()
	mask3dOpPrep.forward(0,pressureDataInit,pressureData)

	#init op
	basicOp=gradio(modelFloat,dataFloat,pressureData)
	op = pyOp.ChainOperator(basicOp,mask3dOp)
	# Outputs
	return modelFloat,dataFloat,pressureData,op

# gradiometry over multiple shots
def gradioOpInitFloat_multi_exp(args):
	"""Function to correctly initialize wave equation operator
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
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis,expAxis])
	dataFloatInit=SepVector.getSepVector(dataHyper,storage="dataFloat")
	#dataFloatInit.scale(0.0)

	# Read In pressure data
	pressureDataFile=parObject.getString("pressureData", "noPressureData")
	if (pressureDataFile == "noPressureData"):
		print("**** WARNING: User did not provide pressureData file. Using zero wfld ****\n")
		pressureDataInit=dataFloatInit.clone()
	else:
		pressureDataInit=genericIO.defaultIO.getVector(pressureDataFile)

	# Read in wavelet and make forcing term
	fullPrior = parObject.getString("fullPrior","none")
	if(fullPrior=="none"):
		print("prior from wavelet")
		_,priorTmp= wriUtilFloat.forcing_term_op_init_m_mutli_exp(args)
		prior=priorTmp.clone()
	else:
		print("full prior")
		priorTmp=genericIO.defaultIO.getVector(fullPrior)
		prior=priorTmp.clone()
	genericIO.defaultIO.writeVector("test_prior.H",prior)
	# calculate data
	laplPressureData = pressureDataInit.clone()
	laplOp = Laplacian2d.laplacian2d_multi_exp(pressureDataInit,laplPressureData)
	if (parObject.getInt("dp",0)==1):
		print("Laplacian DP test:")
		laplOp.dotTest(1)
	laplOp.forward(0,pressureDataInit,laplPressureData)
	dataFloatInit.scale(0.0)
	dataFloatInit.scaleAdd(prior,1,1)
	dataFloatInit.scaleAdd(laplPressureData,1,1)

	#init mask op
	maskWidthSpace = parObject.getInt("maskWidth",15)
	mask4dOp= Mask4d.mask4d(dataFloatInit, dataFloatInit, maskWidthSpace, dataFloatInit.getHyper().axes[0].n-maskWidthSpace, maskWidthSpace,dataFloatInit.getHyper().axes[1].n-maskWidthSpace, 0, dataFloatInit.getHyper().axes[2].n, 0, dataFloatInit.getHyper().axes[3].n, 0)
	mask4dOpPrep= Mask4d.mask4d(dataFloatInit, dataFloatInit,maskWidthSpace, dataFloatInit.getHyper().axes[0].n-maskWidthSpace, maskWidthSpace, dataFloatInit.getHyper().axes[1].n-maskWidthSpace, 0, dataFloatInit.getHyper().axes[2].n, 0, dataFloatInit.getHyper().axes[3].n, 0)
	#mask everything
	dataFloat = dataFloatInit.clone()
	mask4dOp.forward(0,dataFloatInit,dataFloat)
	pressureData = pressureDataInit.clone()
	mask4dOpPrep.forward(0,pressureDataInit,pressureData)

	#init op
	basicOp=gradio_multi_exp(modelFloat,dataFloat,pressureData)
	op = pyOp.ChainOperator(basicOp,mask4dOp)
	# Outputs
	return modelFloat,dataFloat,pressureData,op

# gradiometry over multiple shots with wfld in freq domain
def gradioOpInitFloat_multi_exp_freq(args,pressureData=None):
	"""Function to correctly initialize wave equation operator
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

	freq_np_axis = np.fft.rfftfreq(nts,dts)
	df = freq_np_axis[1]
	nf= freq_np_axis.size
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
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# check input mode
	inputMode=parObject.getString("inputMode","freq")

	# Allocate data
	dataTimeHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis,expAxis])
	wavefieldTimeInit=SepVector.getSepVector(dataTimeHyper,storage ="dataFloat")
	dataFreqHyper=Hypercube.hypercube(axes=[zAxis,xAxis,wAxis,expAxis])
	wavefieldFreqInit=SepVector.getSepVector(dataFreqHyper,storage="dataComplex")
	#dataFloatInit.scale(0.0)

	# init fft op
	fftOpTemp = fft_wfld.fft_wfld(wavefieldFreqInit,wavefieldTimeInit,axis=1)
	if(fmax!=-1.0):
		dataFreqHyper_wind=Hypercube.hypercube(axes=[zAxis,xAxis,wAxis_wind,expAxis])
		wavefieldFreqInit_wind=SepVector.getSepVector(dataFreqHyper_wind,storage="dataComplex")
		padTruncateFreqOp=PadTruncateSource.zero_pad_4d(wavefieldFreqInit_wind,wavefieldFreqInit)
		fftOp=fftOpTemp*padTruncateFreqOp

		wavefieldFreqInit=wavefieldFreqInit_wind
	else:
		fftOp=fftOpTemp

	# Read In pressure data
	pressureDataFile=parObject.getString("pressureData", "noPressureData")
	if (pressureDataFile == "noPressureData"):
		if(pressureData!=None):
			wavefieldFreqInit=pressureData
		else:
			print("**** WARNING: User did not provide pressureData file. Using zero wfld ****")
			wavefieldFreqInit.zero();
	else:
		if (inputMode == 'time'):
		    print('------ input pressure data in time domain. converting to freq ------')
		    wavefieldTimeInit = genericIO.defaultIO.getVector(pressureDataFile)
		    fftOp.adjoint(0,wavefieldFreqInit,wavefieldTimeInit)
		else:
			wavefieldFreqInit=genericIO.defaultIO.getVector(pressureDataFile)

	# Read in wavelet and make forcing term
	fullPrior = parObject.getString("fullPrior","none")
	if(fullPrior=="none"):
		_,priorTime= wriUtilFloat.forcing_term_op_init_m_mutli_exp(args)
		prior=wavefieldFreqInit.clone()
		fftOp.adjoint(0,prior,priorTime) # convert to freq
	else:
		print("full prior")
		priorTmp=genericIO.defaultIO.getVector(fullPrior)
		prior=priorTmp.clone()
	genericIO.defaultIO.writeVector("test_prior.H",prior)
	# calculate data
	laplPressureData = wavefieldFreqInit.clone()
	laplOp = Laplacian2d.Laplacian2d_multi_exp_complex(wavefieldFreqInit,laplPressureData)
	if (parObject.getInt("dp",0)==1):
		print("Laplacian DP test:")
		laplOp.dotTest(1)
	laplOp.forward(0,wavefieldFreqInit,laplPressureData)
	priorPlusLapl=wavefieldFreqInit.clone()
	priorPlusLapl.scale(0.0)
	priorPlusLapl.scaleAdd(prior,1,1)
	priorPlusLapl.scaleAdd(laplPressureData,1,1)

	#init mask op
	fmin=parObject.getFloat("fmin",-1)
	if(fmin==-1):
		freqMask=1
	else:
		freqMask=int(fmin/fftOp.getDomain().getHyper().getAxis(3).d)
	maskWidthSpace = 10
	maskWidthSpacePrep=5
	mask4dOp= Mask4d.mask4d_complex(wavefieldFreqInit, wavefieldFreqInit, maskWidthSpace, wavefieldFreqInit.getHyper().axes[0].n-maskWidthSpace-1, maskWidthSpace,wavefieldFreqInit.getHyper().axes[1].n-maskWidthSpace-1, freqMask, wavefieldFreqInit.getHyper().axes[2].n, 0, wavefieldFreqInit.getHyper().axes[3].n, 0)
	mask4dOpPrep= Mask4d.mask4d_complex(wavefieldFreqInit, wavefieldFreqInit,maskWidthSpacePrep, wavefieldFreqInit.getHyper().axes[0].n-maskWidthSpacePrep, maskWidthSpacePrep, wavefieldFreqInit.getHyper().axes[1].n-maskWidthSpacePrep, freqMask, wavefieldFreqInit.getHyper().axes[2].n, 0, wavefieldFreqInit.getHyper().axes[3].n, 0)
	#mask everything
	data = priorPlusLapl.clone()
	mask4dOp.forward(0,priorPlusLapl,data)
	pressureData = wavefieldFreqInit.clone()
	mask4dOpPrep.forward(0,wavefieldFreqInit,pressureData)

	#init op
	basicOp=gradio_multi_exp_freq(modelFloat,data,pressureData)
	op = pyOp.ChainOperator(basicOp,mask4dOp)
	# Outputs
	return modelFloat,data,pressureData,op,fftOp

def gradioOpInitFloat_givenPressure(pressureData,args):
	"""Function to correctly initialize wave equation operator
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	parObject=genericIO.io(params=sys.argv)

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
	modelHyper=Hypercube.hypercube(axes=[zAxis,xAxis])
	modelFloat=SepVector.getSepVector(modelHyper,storage="dataFloat")

	# Allocate data
	dataHyper=Hypercube.hypercube(axes=[zAxis,xAxis,timeAxis])
	dataFloatInit=SepVector.getSepVector(dataHyper,storage="dataFloat")
	dataFloatInit.scale(0.0)

	# Read In pressure data
	pressureDataInit = pressureData

	# Read in wavelet and make forcing term
	fullPrior = parObject.getString("fullPrior","none")
	if(fullPrior=="none"):
		print("prior from wavelet")
		_,priorTmp= wriUtilFloat.forcing_term_op_init_m(args[:])
		prior=priorTmp.clone()
	else:
		print("full prior")
		priorTmp=genericIO.defaultIO.getVector(fullPrior)
		prior=priorTmp.clone()

	# calculate data
	laplPressureData = pressureDataInit.clone()
	laplOp = Laplacian2d.laplacian2d(pressureDataInit,laplPressureData)
	if (parObject.getInt("dp",0)==1):
		print("Laplacian DP test:")
		laplOp.dotTest(1)
	laplOp.forward(0,pressureDataInit,laplPressureData)
	dataFloatInit.scale(0.0)
	dataFloatInit.scaleAdd(prior,1,1)
	dataFloatInit.scaleAdd(laplPressureData,1,1)

	#init mask op
	maskWidthSpace = parObject.getInt("maskWidth",15)
	mask3dOp= Mask3d.mask3d(dataFloatInit,dataFloatInit,maskWidthSpace,dataFloatInit.getHyper().axes[0].n-maskWidthSpace,maskWidthSpace,dataFloatInit.getHyper().axes[1].n-maskWidthSpace,0,dataFloatInit.getHyper().axes[2].n,0)
	mask3dOpPrep= Mask3d.mask3d(dataFloatInit,dataFloatInit,maskWidthSpace,dataFloatInit.getHyper().axes[0].n-maskWidthSpace,maskWidthSpace,dataFloatInit.getHyper().axes[1].n-maskWidthSpace,0,dataFloatInit.getHyper().axes[2].n,0)
	#mask everything
	dataFloat = dataFloatInit.clone()
	mask3dOp.forward(0,dataFloatInit,dataFloat)
	pressureData = pressureDataInit.clone()
	mask3dOpPrep.forward(0,pressureDataInit,pressureData)

	#init op
	basicOp=gradio(modelFloat,dataFloat,pressureData)
	op = pyOp.ChainOperator(basicOp,mask3dOp)
	# Outputs
	return modelFloat,dataFloat,pressureData,op

def update_data(newPressureData,args):

	parObject=genericIO.io(params=sys.argv)

	pressureDataInit=newPressureData

	# Read in wavelet and make forcing term
	_,priorTmp= wriUtilFloat.forcing_term_op_init_m(args)
	prior=priorTmp.clone()

	# calculate data
	laplPressureData = pressureDataInit.clone()
	laplOp = Laplacian2d.laplacian2d(pressureDataInit,laplPressureData)
	if (parObject.getInt("dp",0)==1):
		print("Laplacian DP test:")
		laplOp.dotTest(1)
	laplOp.forward(0,pressureDataInit,laplPressureData)
	dataFloatInit=pressureDataInit.clone()
	dataFloatInit.scale(0.0)
	dataFloatInit.scaleAdd(prior,1,1)
	dataFloatInit.scaleAdd(laplPressureData,1,1)

	#init mask op
	maskWidthSpace = parObject.getInt("maskWidth",0)
	mask3dOp= Mask3d.mask3d(dataFloatInit,dataFloatInit,maskWidthSpace,dataFloatInit.getHyper().axes[0].n-maskWidthSpace,maskWidthSpace,dataFloatInit.getHyper().axes[1].n-maskWidthSpace,0,dataFloatInit.getHyper().axes[2].n,0)
	mask3dOpPrep= Mask3d.mask3d(dataFloatInit,dataFloatInit,maskWidthSpace,dataFloatInit.getHyper().axes[0].n-maskWidthSpace,maskWidthSpace,dataFloatInit.getHyper().axes[1].n-maskWidthSpace,0,dataFloatInit.getHyper().axes[2].n,0)
	#mask everything
	dataFloat = dataFloatInit.clone()
	mask3dOp.forward(0,dataFloatInit,dataFloat)
	pressureData = pressureDataInit.clone()
	mask3dOpPrep.forward(0,pressureDataInit,pressureData)

	# Outputs
	return dataFloat,pressureData

class gradio(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,pressureData):
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		if("getCpp" in dir(pressureData)):
			pressureData = pressureData.getCpp()
		self.pyOp = pyGradio.Gradio(domain,range,pressureData)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyGradio.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyGradio.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyGradio.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result
	def update_wfld(self,new_wfld):
		if("getCpp" in dir(new_wfld)):
			new_wfld = new_wfld.getCpp()
		with pyGradio.ostream_redirect():
			self.pyOp.set_wfld(new_wfld)
		return

class gradio_multi_exp(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,pressureData):
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		if("getCpp" in dir(pressureData)):
			pressureData = pressureData.getCpp()
		self.pyOp = pyGradio.Gradio_multi_exp(domain,range,pressureData)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyGradio.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyGradio.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyGradio.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result
	def update_wfld(self,new_wfld):
		if("getCpp" in dir(new_wfld)):
			new_wfld = new_wfld.getCpp()
		with pyGradio.ostream_redirect():
			self.pyOp.set_wfld(new_wfld)
		return

class gradio_multi_exp_freq(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for acoustic wave equation"""

	def __init__(self,domain,range,pressureData):
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(range)):
			range = range.getCpp()
		if("getCpp" in dir(pressureData)):
			pressureData = pressureData.getCpp()
		self.pyOp = pyGradio.Gradio_multi_exp_freq(domain,range,pressureData)
		return

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyGradio.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyGradio.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyGradio.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result
	def update_wfld(self,new_wfld):
		if("getCpp" in dir(new_wfld)):
			new_wfld = new_wfld.getCpp()
		with pyGradio.ostream_redirect():
			self.pyOp.set_wfld(new_wfld)
		return

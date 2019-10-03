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

def gradioOpInitFloat(args):
	"""Function to correctly initialize wave equation operator
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

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

def gradioOpInitFloat_givenPressure(pressureData,args):
	"""Function to correctly initialize wave equation operator
	   The function will return the necessary variables for operator construction
	"""
	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args[:])
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

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

	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

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




#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyInterpBSpline2dDouble
import genericIO
import SepVector
import Hypercube
import numpy as np
import math

def generateSplineMesh1d(positions,sampling,mesh,tolerance):

	"""Function that creates a mesh for spline interpolation using B-splines"""

	if (mesh == "reg"):

		# Read sampling parameters
		splineMesh=[]
		oMesh=positions[0]
		fMesh=positions[len(positions)-1]
		dMesh=sampling[0]
		diff=fMesh-oMesh
		nMesh=diff/dMesh
		nMesh=int(round(nMesh))+1
		# Generate mesh
		for iPos in range(nMesh):
			pos=oMesh+iPos*dMesh
			splineMesh.append(pos)

	elif (mesh == "irreg"):

		# Number of knots
		nPoint=len(positions)

		# Read parameters
		splineMesh=[]
		oMesh=positions[0]
		fMesh=positions[nPoint-1]

		# Loop over knots
		for iPoint in range(nPoint-1):

			# Compute the position for that knot
			pos=positions[iPoint]
			while ( pos < positions[iPoint+1] and abs(pos-positions[iPoint+1]) > tolerance ):
				splineMesh.append(pos)
				pos=pos+sampling[iPoint]

		splineMesh.append(fMesh)

	splineMeshNp=np.asarray(splineMesh)

	return splineMeshNp

# Create an initializer
def bSpline2dDoubleInit(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Interpolation parameters
	zOrder=parObject.getInt("zOrder",3)
	xOrder=parObject.getInt("xOrder",3)
	nzParam=parObject.getInt("nzParam",1)
	nxParam=parObject.getInt("nxParam",1)
	scaling=parObject.getInt("scaling",1)
	zTolerance=parObject.getFloat("zTolerance",0.25)
	xTolerance=parObject.getFloat("xTolerance",0.25)
	fat=parObject.getInt("fat",5)

	# Read data positions
	dataFile=parObject.getString("vel")
	dataFile=genericIO.defaultIO.getVector(dataFile)

	# z-axis
	zDataAxis=dataFile.getHyper().axes[0]
	dzData=zDataAxis.d
	nzData=zDataAxis.n
	ozData=zDataAxis.o

	# x-axis
	xDataAxis=dataFile.getHyper().axes[1]
	dxData=xDataAxis.d
	nxData=xDataAxis.n
	oxData=xDataAxis.o

	# Mesh for both directions
	zMeshFile=parObject.getString("zMesh","noZMeshFile")
	xMeshFile=parObject.getString("xMesh","noXMeshFile")

	# Case where user does not provide a z-mesh
	if (zMeshFile=="noZMeshFile"):

		# Compute mesh bounds (mesh should not include the fat layer)
		ozMesh=zDataAxis.o+fat*dzData
		fzMesh=zDataAxis.o+(nzData-fat-1)*dzData

		# Get mesh parameters
		zPositions=parObject.getFloats("zPositions",[])
		zPositions.insert(0,ozMesh)
		zPositions.append(fzMesh)
		zSampling=parObject.getFloats("zSampling",[])
		zMesh=parObject.getString("zMeshType","reg")

		# Create mesh and convert to double1DReg
		zMeshTolerance=zTolerance*dzData
		zSplineMeshNp=generateSplineMesh1d(zPositions,zSampling,zMesh,zMeshTolerance)
		zMeshAxis=Hypercube.axis(n=zSplineMeshNp.size)
		zMeshHyper=Hypercube.hypercube(axes=[zMeshAxis])
		zSplineMeshDouble=SepVector.getSepVector(zMeshHyper,storage="dataDouble")
		zSplineMeshDoubleNp=zSplineMeshDouble.getNdArray()
		zSplineMeshDoubleNp[:]=zSplineMeshNp

	# Case where user provides the z-mesh
	else:

		# Read and create mesh
		zSplineMeshFloat=genericIO.defaultIO.getVector(zMeshFile)
		zSplineMeshFloatNp=zSplineMeshFloat.getNdArray()
		zSplineMeshDouble=SepVector.getSepVector(zSplineMeshFloat.getHyper(),storage="dataDouble")
		zSplineMeshDoubleNp=zSplineMeshDouble.getNdArray()
		zSplineMeshDoubleNp[:]=zSplineMeshFloatNp
		zMeshAxis=Hypercube.axis(n=zSplineMeshDoubleNp.size)
		zMeshHyper=Hypercube.hypercube(axes=[zMeshAxis])

	if (xMeshFile=="noXMeshFile"):

		# Compute mesh bounds (mesh should not include the fat layer)
		oxMesh=xDataAxis.o+fat*dxData
		fxMesh=xDataAxis.o+(nxData-fat-1)*dxData

		# Get mesh parameters
		xPositions=parObject.getFloats("xPositions",[])
		xPositions.insert(0,oxMesh)
		xPositions.append(fxMesh)
		xSampling=parObject.getFloats("xSampling",[])
		xMesh=parObject.getString("xMeshType","reg")

		# Create mesh and convert to double1DReg
		xMeshTolerance=xTolerance*dxData
		xSplineMeshNp=generateSplineMesh1d(xPositions,xSampling,xMesh,xMeshTolerance)
		xMeshAxis=Hypercube.axis(n=xSplineMeshNp.size)
		xMeshHyper=Hypercube.hypercube(axes=[xMeshAxis])
		xSplineMeshDouble=SepVector.getSepVector(xMeshHyper,storage="dataDouble")
		xSplineMeshDoubleNp=xSplineMeshDouble.getNdArray()
		xSplineMeshDoubleNp[:]=xSplineMeshNp

	# Case where user provides the z-mesh
	else:

		# Read and create mesh
		xSplineMeshFloat=genericIO.defaultIO.getVector(xMeshFile)
		xSplineMeshFloatNp=xSplineMeshFloat.getNdArray()
		xSplineMeshDouble=SepVector.getSepVector(xSplineMeshFloat.getHyper(),storage="dataDouble")
		xSplineMeshDoubleNp=xSplineMeshDouble.getNdArray()
		xSplineMeshDoubleNp[:]=xSplineMeshFloatNp
		xMeshAxis=Hypercube.axis(n=xSplineMeshDoubleNp.size)
		xMeshHyper=Hypercube.hypercube(axes=[xMeshAxis])


	# Allocate model and fill with zeros
	modelHyper=Hypercube.hypercube(axes=[zMeshAxis,xMeshAxis])
	modelDouble=SepVector.getSepVector(modelHyper,storage="dataDouble")

	# Allocate data and fill with zeros
	dataHyper=Hypercube.hypercube(axes=[zDataAxis,xDataAxis])
	dataDouble=SepVector.getSepVector(dataHyper,storage="dataDouble")

	return modelDouble,dataDouble,zOrder,xOrder,zSplineMeshDouble,xSplineMeshDouble,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat


class bSpline2dDouble(Op.Operator):
	def __init__(self,domain,range,zOrder,xOrder,zControlPoints,xControlPoints,zDataAxis,xDataAxis,zParam,xParam,scaling,zTolerance,xTolerance,fat,buildParamVector):
		self.setDomainRange(domain,range)
		if("getCpp" in dir(zControlPoints)):
			zControlPoints = zControlPoints.getCpp()
		if("getCpp" in dir(xControlPoints)):
			xControlPoints = xControlPoints.getCpp()
		if("getCpp" in dir(zDataAxis)):
			zDataAxis = zDataAxis.getCpp()
		if("getCpp" in dir(xDataAxis)):
			xDataAxis = xDataAxis.getCpp()
		if("getCpp" in dir(zParam)):
				zParam = zParam.getCpp()
		if("getCpp" in dir(xParam)):
				xParam = xParam.getCpp()
		if (buildParamVector==1):
			# Use contructor that builds the parameter vectors
			self.pyOp = pyInterpBSpline2dDouble.interpBSpline2dDouble(zOrder,xOrder,zControlPoints,xControlPoints,zDataAxis,xDataAxis,zParam,xParam,scaling,zTolerance,xTolerance,fat)
		else:
			# Use contructor that builds the parameter vectors
			self.pyOp = pyInterpBSpline2dDouble.interpBSpline2dDouble(zOrder,xOrder,zControlPoints,xControlPoints,zDataAxis,xDataAxis,zParam,xParam,scaling,zTolerance,xTolerance,fat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline2dDouble.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline2dDouble.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def getZParamVector(self):
		with pyInterpBSpline2dDouble.ostream_redirect():
			zParam=self.pyOp.getZParamVector()
			zParam=SepVector.doubleVector(fromCpp=zParam)
		return zParam

	def getXParamVector(self):
		with pyInterpBSpline2dDouble.ostream_redirect():
			xParam=self.pyOp.getXParamVector()
			xParam=SepVector.doubleVector(fromCpp=xParam)
		return xParam

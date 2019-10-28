#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyInterpSplineInv
import genericIO
import SepVector
import Hypercube
import numpy as np
import math

# Create an initializer
def bSpline2dInit(args):

	# IO object
	parObject=genericIO.io(params=args)

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
	fzData=ozData+(nzData-1)*dzData

	# x-axis
	xDataAxis=dataFile.getHyper().axes[1]
	dxData=xDataAxis.d
	nxData=xDataAxis.n
	oxData=xDataAxis.o
	fzData=ozData+(nzData-1)*dzData

	# Mesh for both directions
	zMeshFile=parObject.getString("zMesh","noZMeshFile")
	xMeshFile=parObject.getString("xMesh","noXMeshFile")

	# Error tolerance
	zMeshTolerance=zTolerance*dzData
	xMeshTolerance=xTolerance*dxData

	# Compute mesh bounds (mesh should not include the fat layer)
	ozMesh=zDataAxis.o+fat*dzData
	fzMesh=zDataAxis.o+(nzData-fat-1)*dzData
	oxMesh=xDataAxis.o+fat*dxData
	fxMesh=xDataAxis.o+(nxData-fat-1)*dxData

	# Case where user does not provide a z-mesh
	if (zMeshFile=="noZMeshFile"):

		# Get mesh parameters
		zPositions=parObject.getFloats("zPositions",[])
		zPositions.insert(0,ozMesh)
		zPositions.append(fzMesh)
		zSampling=parObject.getFloats("zSampling",[])
		zMesh=parObject.getString("zMeshType","irreg")

		# Create mesh
		zSplineMeshNpTemp=generateSplineMesh1d(zPositions,zSampling,zMesh,zMeshTolerance)
		zMeshAxis=Hypercube.axis(n=zSplineMeshNpTemp.size)
		zMeshHyper=Hypercube.hypercube(axes=[zMeshAxis])
		zSplineMesh=SepVector.getSepVector(zMeshHyper)
		zSplineMeshNp=zSplineMesh.getNdArray()
		zSplineMeshNp[:]=zSplineMeshNpTemp

	# Case where user provides the z-mesh
	else:

		# Read and create mesh
		zSplineMesh=genericIO.defaultIO.getVector(zMeshFile)
		zMeshAxis=Hypercube.axis(n=zSplineMesh.getHyper().axes[0].n)

	if (xMeshFile=="noXMeshFile"):

		# Get mesh parameters
		xPositions=parObject.getFloats("xPositions",[])
		xPositions.insert(0,oxMesh)
		xPositions.append(fxMesh)
		xSampling=parObject.getFloats("xSampling",[])
		xMesh=parObject.getString("xMeshType","irreg")

		# Create mesh and convert to double1DReg
		xSplineMeshNpTemp=generateSplineMesh1d(xPositions,xSampling,xMesh,xMeshTolerance)
		xMeshAxis=Hypercube.axis(n=xSplineMeshNpTemp.size)
		xMeshHyper=Hypercube.hypercube(axes=[xMeshAxis])
		xSplineMesh=SepVector.getSepVector(xMeshHyper)
		xSplineMeshNp=xSplineMesh.getNdArray()
		xSplineMeshNp[:]=xSplineMeshNpTemp

	# Case where user provides the x-mesh
	else:

		# Read and create mesh
		xSplineMesh=genericIO.defaultIO.getVector(xMeshFile)
		xMeshAxis=Hypercube.axis(n=xSplineMesh.getHyper().axes[0].n)

	# Check that the mesh initial and final values coincide with data bounds
	zSplineMeshNp=zSplineMesh.getNdArray()
	xSplineMeshNp=xSplineMesh.getNdArray()

	# zMesh
	ozMeshOut=zSplineMeshNp[0]
	fzMeshOut=zSplineMeshNp[zSplineMeshNp.size-1]
	if ( abs(ozMeshOut-ozMesh) > zMeshTolerance or abs(fzMeshOut-fzMesh) > zMeshTolerance ):
		print("**** ERROR [zMesh]: zMesh start/end points do not coincide with data grid ****")

	# xMesh
	oxMeshOut=xSplineMeshNp[0]
	fxMeshOut=xSplineMeshNp[xSplineMeshNp.size-1]
	if ( abs(oxMeshOut-oxMesh) > xMeshTolerance or abs(fxMeshOut-fxMesh) > xMeshTolerance ):
		print("**** ERROR [xMesh]: xMesh start/end points do not coincide with data grid ****")

	# Allocate model and fill with zeros
	modelHyper=Hypercube.hypercube(axes=[zMeshAxis,xMeshAxis])
	model=SepVector.getSepVector(modelHyper)

	# Allocate data and fill with zeros
	dataHyper=Hypercube.hypercube(axes=[zDataAxis,xDataAxis])
	data=SepVector.getSepVector(dataHyper)

	return model,data,zOrder,xOrder,zSplineMesh,xSplineMesh,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat


class bSpline2d(Op.Operator):
	def __init__(self,domain,range,zOrder,xOrder,zControlPoints,xControlPoints,zDataAxis,xDataAxis,zParam,xParam,scaling,zTolerance,xTolerance,fat):
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
		self.pyOp = pyInterpBSpline2d.interpBSpline2d(zOrder,xOrder,zControlPoints,xControlPoints,zDataAxis,xDataAxis,zParam,xParam,scaling,zTolerance,xTolerance,fat)

		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline2d.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline2d.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
			return

	def getZParamVector(self):
		with pyInterpBSpline2d.ostream_redirect():
			zParam=self.pyOp.getZParamVector()
			zParam=SepVector.floatVector(fromCpp=zParam)
		return zParam

	def getXParamVector(self):
		with pyInterpBSpline2d.ostream_redirect():
			xParam=self.pyOp.getXParamVector()
			xParam=SepVector.floatVector(fromCpp=xParam)
		return xParam

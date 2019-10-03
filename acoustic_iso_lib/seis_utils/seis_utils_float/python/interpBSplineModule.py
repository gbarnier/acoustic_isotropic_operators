#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyInterpBSpline1d
import pyInterpBSpline2d
import pyInterpBSpline3d
import genericIO
import SepVector
import Hypercube
import numpy as np
import math

# Generate 1D spline mesh
def generateSplineMesh1d(positions,sampling,sub,mesh,tolerance,nDataNf):

	"""Function that creates a mesh for spline interpolation using B-splines"""

	# Regular mesh
	if (mesh == "reg"):

		# Read sampling parameters
		splineMesh=[]
		oMesh=positions[0]
		fMesh=positions[len(positions)-1]
		nMesh=int(nDataNf/sub)
		dMesh=(fMesh-oMesh)/(nMesh-1)

		# Generate mesh
		for iPos in range(nMesh):
			pos=oMesh+iPos*dMesh
			splineMesh.append(pos)

	# Irregular mesh
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

# 1d spline
def bSpline1dInit(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Interpolation parameters
	zOrder=parObject.getInt("zOrder",3)
	nzParam=parObject.getInt("nzParam",30000)
	scaling=parObject.getInt("scaling",1)
	zTolerance=parObject.getFloat("zTolerance",0.25)
	fat=parObject.getInt("fat",5)
	zSub=parObject.getInt("zSub",1)
	# Read data positions
	dataFile=parObject.getString("vel")
	dataFile=genericIO.defaultIO.getVector(dataFile)

	# z-axis
	zDataAxis=dataFile.getHyper().axes[0]
	dzData=zDataAxis.d
	nzData=zDataAxis.n
	ozData=zDataAxis.o
	fzData=ozData+(nzData-1)*dzData

	# Mesh for both directions
	zMeshFile=parObject.getString("zMeshIn","noZMeshFile")

	# Error tolerance
	zMeshTolerance=zTolerance*dzData

	# Compute mesh bounds (mesh should not include the fat layer)
	ozMesh=zDataAxis.o+fat*dzData
	fzMesh=zDataAxis.o+(nzData-fat-1)*dzData

	# Number of data points without the fat
	nzDataNf=nzData-2*fat

	# Case where user does not provide a z-mesh
	if (zMeshFile=="noZMeshFile"):

		# Get mesh parameters
		zPositions=parObject.getFloats("zPositions",[])
		zPositions.insert(0,ozMesh)
		zPositions.append(fzMesh)
		zSampling=parObject.getFloats("zSampling",[])
		zMesh=parObject.getString("zMeshType","irreg")

		# Create mesh
		zSplineMeshNpTemp=generateSplineMesh1d(zPositions,zSampling,zSub,zMesh,zMeshTolerance,nzDataNf)
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

	# Check that the mesh initial and final values coincide with data bounds
	zSplineMeshNp=zSplineMesh.getNdArray()

	# zMesh
	ozMeshOut=zSplineMeshNp[0]
	fzMeshOut=zSplineMeshNp[zSplineMeshNp.size-1]
	if ( abs(ozMeshOut-ozMesh) > zMeshTolerance or abs(fzMeshOut-fzMesh) > zMeshTolerance ):
		print("**** ERROR [zMesh]: zMesh start/end points do not coincide with data grid ****")

	# Allocate model and fill with zeros
	modelHyper=Hypercube.hypercube(axes=[zMeshAxis])
	model=SepVector.getSepVector(modelHyper)

	# Allocate data and fill with zeros
	dataHyper=Hypercube.hypercube(axes=[zDataAxis])
	data=SepVector.getSepVector(dataHyper)

	return model,data,zOrder,zSplineMesh,zDataAxis,nzParam,scaling,zTolerance,fat
class bSpline1d(Op.Operator):
	def __init__(self,domain,range,zOrder,zModel,zDataAxis,nzParam,scaling,zTolerance,fat):
		self.setDomainRange(domain,range)
		if("getCpp" in dir(zModel)):
			zModel = zModel.getCpp()
		if("getCpp" in dir(zDataAxis)):
			zDataAxis = zDataAxis.getCpp()
		self.pyOp = pyInterpBSpline1d.interpBSpline1d(zOrder,zModel,zDataAxis,nzParam,scaling,zTolerance,fat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline1d.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline1d.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
			return

	def getZMeshModel(self):
		with pyInterpBSpline1d.ostream_redirect():
			zMeshVector=self.pyOp.getZMeshModel()
			zMeshVector=SepVector.floatVector(fromCpp=zMeshVector)
		return zMeshVector

	def getZMeshModel1d(self):
		with pyInterpBSpline1d.ostream_redirect():
			zMeshVector=self.pyOp.getZControlPoints()
			zMeshVector=SepVector.floatVector(fromCpp=zMeshVector)
		return zMeshVector

	def getZMeshData(self):
		with pyInterpBSpline1d.ostream_redirect():
			zMeshVector=self.pyOp.getZMeshData()
			zMeshVector=SepVector.floatVector(fromCpp=zMeshVector)
		return zMeshVector

# 2d spline
def bSpline2dInit(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Interpolation parameters
	zOrder=parObject.getInt("zOrder",3)
	xOrder=parObject.getInt("xOrder",3)
	nzParam=parObject.getInt("nzParam",50000)
	nxParam=parObject.getInt("nxParam",50000)
	scaling=parObject.getInt("scaling",1)
	zTolerance=parObject.getFloat("zTolerance",0.25)
	xTolerance=parObject.getFloat("xTolerance",0.25)
	fat=parObject.getInt("fat",5)
	zSub=parObject.getInt("zSub",1)
	xSub=parObject.getInt("xSub",1)

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
	fxData=oxData+(nxData-1)*dxData

	# Mesh for both directions
	zMeshFile=parObject.getString("zMeshIn","noZMeshFile")
	xMeshFile=parObject.getString("xMeshIn","noXMeshFile")

	# Error tolerance
	zMeshTolerance=zTolerance*dzData
	xMeshTolerance=xTolerance*dxData

	# Compute mesh bounds (mesh should not include the fat layer)
	ozMesh=zDataAxis.o+fat*dzData
	fzMesh=zDataAxis.o+(nzData-fat-1)*dzData
	oxMesh=xDataAxis.o+fat*dxData
	fxMesh=xDataAxis.o+(nxData-fat-1)*dxData

	# Number of data points without the fat
	nzDataNf=nzData-2*fat
	nxDataNf=nxData-2*fat

	# Case where user does not provide a z-mesh
	if (zMeshFile=="noZMeshFile"):

		# Get mesh parameters
		zPositions=parObject.getFloats("zPositions",[])
		zPositions.insert(0,ozMesh)
		zPositions.append(fzMesh)
		zSampling=parObject.getFloats("zSampling",[])
		zMesh=parObject.getString("zMeshType","irreg")

		# Create mesh
		zSplineMeshNpTemp=generateSplineMesh1d(zPositions,zSampling,zSub,zMesh,zMeshTolerance,nzDataNf)
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
		xSplineMeshNpTemp=generateSplineMesh1d(xPositions,xSampling,xSub,xMesh,xMeshTolerance,nxDataNf)
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
	def __init__(self,domain,range,zOrder,xOrder,zControlPoints,xControlPoints,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat):
		self.setDomainRange(domain,range)
		if("getCpp" in dir(zControlPoints)):
			zControlPoints = zControlPoints.getCpp()
		if("getCpp" in dir(xControlPoints)):
			xControlPoints = xControlPoints.getCpp()
		if("getCpp" in dir(zDataAxis)):
			zDataAxis = zDataAxis.getCpp()
		if("getCpp" in dir(xDataAxis)):
			xDataAxis = xDataAxis.getCpp()
		self.pyOp = pyInterpBSpline2d.interpBSpline2d(zOrder,xOrder,zControlPoints,xControlPoints,zDataAxis,xDataAxis,nzParam,nxParam,scaling,zTolerance,xTolerance,fat)

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

	def getZMeshModel(self):
		with pyInterpBSpline2d.ostream_redirect():
			zMeshVector=self.pyOp.getZMeshModel()
			zMeshVector=SepVector.floatVector(fromCpp=zMeshVector)
		return zMeshVector

	def getZMeshModel1d(self):
		with pyInterpBSpline2d.ostream_redirect():
			zMeshVector=self.pyOp.getZControlPoints()
			zMeshVector=SepVector.floatVector(fromCpp=zMeshVector)
		return zMeshVector

	def getZMeshData(self):
		with pyInterpBSpline2d.ostream_redirect():
			zMeshDataVector=self.pyOp.getZMeshData()
			zMeshDataVector=SepVector.floatVector(fromCpp=zMeshDataVector)
		return zMeshDataVector

	def getXMeshModel(self):
		with pyInterpBSpline2d.ostream_redirect():
			xMeshVector=self.pyOp.getXMeshModel()
			xMeshVector=SepVector.floatVector(fromCpp=xMeshVector)
		return xMeshVector

	def getXMeshModel1d(self):
		with pyInterpBSpline2d.ostream_redirect():
			xMeshVector=self.pyOp.getXControlPoints()
			xMeshVector=SepVector.floatVector(fromCpp=xMeshVector)
		return xMeshVector

	def getXMeshData(self):
		with pyInterpBSpline2d.ostream_redirect():
			xMeshDataVector=self.pyOp.getXMeshData()
			xMeshDataVector=SepVector.floatVector(fromCpp=xMeshDataVector)
		return xMeshDataVector

# 3d spline
def bSpline3dInit(args):

	# IO object
	parObject=genericIO.io(params=sys.argv)

	# Interpolation parameters
	zOrder=parObject.getInt("zOrder",3)
	xOrder=parObject.getInt("xOrder",3)
	yOrder=parObject.getInt("yOrder",3)
	nzParam=parObject.getInt("nzParam",30000)
	nxParam=parObject.getInt("nxParam",50000)
	nyParam=parObject.getInt("nyParam",30000)
	scaling=parObject.getInt("scaling",1)
	zTolerance=parObject.getFloat("zTolerance",0.25)
	xTolerance=parObject.getFloat("xTolerance",0.25)
	yTolerance=parObject.getFloat("yTolerance",0.25)
	zFat=parObject.getInt("zFat",5)
	xFat=parObject.getInt("xFat",5)
	yFat=parObject.getInt("yFat",5)
	zSub=parObject.getInt("zSub",1)
	xSub=parObject.getInt("xSub",1)
	ySub=parObject.getInt("ySub",1)

	# Read data positions
	dataFile=parObject.getString("vel")
	dataFile=genericIO.defaultIO.getVector(dataFile)
	dataHyper=dataFile.getHyper()
	if (dataHyper.getNdim() < 3):
		nExt=parObject.getInt("nExt")
		extension=parObject.getString("extension")
		if (extension=="time"):	dExt=parObject.getFloat("dts")
		if (extension=="offset"): dExt=parObject.getFloat("dx")
		oExt=-(nExt-1)/2*dExt
		extAxis=Hypercube.axis(n=nExt,o=oExt,d=dExt)
		dataHyper.addAxis(extAxis)
	dataFile=SepVector.getSepVector(dataHyper)

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
	fxData=oxData+(nxData-1)*dxData

	# y-axis
	yDataAxis=dataFile.getHyper().axes[2]
	dyData=yDataAxis.d
	nyData=yDataAxis.n
	oyData=yDataAxis.o
	fyData=ozData+(nzData-1)*dzData

	# Mesh for both directions
	zMeshFile=parObject.getString("zMeshIn","noZMeshFile")
	xMeshFile=parObject.getString("xMeshIn","noXMeshFile")
	yMeshFile=parObject.getString("yMeshIn","noYMeshFile")

	# Error tolerance
	zMeshTolerance=zTolerance*dzData
	xMeshTolerance=xTolerance*dxData
	yMeshTolerance=yTolerance*dyData

	# Compute mesh bounds (mesh should not include the fat layer)
	ozMesh=zDataAxis.o+zFat*dzData
	fzMesh=zDataAxis.o+(nzData-zFat-1)*dzData
	oxMesh=xDataAxis.o+xFat*dxData
	fxMesh=xDataAxis.o+(nxData-xFat-1)*dxData
	oyMesh=yDataAxis.o+yFat*dyData
	fyMesh=yDataAxis.o+(nyData-yFat-1)*dyData

	# Number of data points without the fat
	nzDataNf=nzData-2*zFat
	nxDataNf=nxData-2*xFat
	nyDataNf=nyData-2*yFat

	# Case where user does not provide a z-mesh
	if (zMeshFile=="noZMeshFile"):

		# Get mesh parameters
		zPositions=parObject.getFloats("zPositions",[])
		zPositions.insert(0,ozMesh)
		zPositions.append(fzMesh)
		zSampling=parObject.getFloats("zSampling",[])
		zMesh=parObject.getString("zMeshType","irreg")
		# Create mesh
		zSplineMeshNpTemp=generateSplineMesh1d(zPositions,zSampling,zSub,zMesh,zMeshTolerance,nzDataNf)
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
		xSplineMeshNpTemp=generateSplineMesh1d(xPositions,xSampling,xSub,xMesh,xMeshTolerance,nxDataNf)
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

	if (yMeshFile=="noYMeshFile"):

		# Get mesh parameters
		yPositions=parObject.getFloats("yPositions",[])
		yPositions.insert(0,oyMesh)
		yPositions.append(fyMesh)
		ySampling=parObject.getFloats("ySampling",[])
		yMesh=parObject.getString("yMeshType","irreg")

		# Create mesh and convert to double1DReg
		ySplineMeshNpTemp=generateSplineMesh1d(yPositions,ySampling,ySub,yMesh,yMeshTolerance,nyDataNf)
		yMeshAxis=Hypercube.axis(n=ySplineMeshNpTemp.size)
		yMeshHyper=Hypercube.hypercube(axes=[yMeshAxis])
		ySplineMesh=SepVector.getSepVector(yMeshHyper)
		ySplineMeshNp=ySplineMesh.getNdArray()
		ySplineMeshNp[:]=ySplineMeshNpTemp

	# Case where user provides the y-mesh
	else:

		# Read and create mesh
		ySplineMesh=genericIO.defaultIO.getVector(yMeshFile)
		yMeshAxis=Hypercube.axis(n=ySplineMesh.getHyper().axes[0].n)

	# Check that the mesh initial and final values coincide with data bounds
	zSplineMeshNp=zSplineMesh.getNdArray()
	xSplineMeshNp=xSplineMesh.getNdArray()
	ySplineMeshNp=ySplineMesh.getNdArray()

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

	# yMesh
	oyMeshOut=ySplineMeshNp[0]
	fyMeshOut=ySplineMeshNp[ySplineMeshNp.size-1]
	if ( abs(oyMeshOut-oyMesh) > yMeshTolerance or abs(fyMeshOut-fyMesh) > yMeshTolerance ):
		print("**** ERROR [xMesh]: yMesh start/end points do not coincide with data grid ****")

	# Allocate model and fill with zeros
	modelHyper=Hypercube.hypercube(axes=[zMeshAxis,xMeshAxis,yMeshAxis])
	model=SepVector.getSepVector(modelHyper)

	# Allocate data and fill with zeros
	dataHyper=Hypercube.hypercube(axes=[zDataAxis,xDataAxis,yDataAxis])
	data=SepVector.getSepVector(dataHyper)

	return model,data,zOrder,xOrder,yOrder,zSplineMesh,xSplineMesh,ySplineMesh,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat

class bSpline3d(Op.Operator):
	def __init__(self,domain,range,zOrder,xOrder,yOrder,zControlPoints,xControlPoints,yControlPoints,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat):
		self.setDomainRange(domain,range)
		if("getCpp" in dir(zControlPoints)):
			zControlPoints = zControlPoints.getCpp()
		if("getCpp" in dir(xControlPoints)):
			xControlPoints = xControlPoints.getCpp()
		if("getCpp" in dir(yControlPoints)):
			yControlPoints = yControlPoints.getCpp()
		if("getCpp" in dir(zDataAxis)):
			zDataAxis = zDataAxis.getCpp()
		if("getCpp" in dir(xDataAxis)):
			xDataAxis = xDataAxis.getCpp()
		if("getCpp" in dir(yDataAxis)):
			yDataAxis = yDataAxis.getCpp()

		self.pyOp = pyInterpBSpline3d.interpBSpline3d(zOrder,xOrder,yOrder,zControlPoints,xControlPoints,yControlPoints,zDataAxis,xDataAxis,yDataAxis,nzParam,nxParam,nyParam,scaling,zTolerance,xTolerance,yTolerance,zFat,xFat,yFat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline3d.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline3d.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
			return

	def getZMeshModel(self):
		with pyInterpBSpline3d.ostream_redirect():
			zMeshVector=self.pyOp.getZMeshModel()
			zMeshVector=SepVector.floatVector(fromCpp=zMeshVector)
		return zMeshVector

	def getXMeshModel(self):
		with pyInterpBSpline3d.ostream_redirect():
			xMeshVector=self.pyOp.getXMeshModel()
			xMeshVector=SepVector.floatVector(fromCpp=xMeshVector)
		return xMeshVector

	def getYMeshModel(self):
		with pyInterpBSpline3d.ostream_redirect():
			yMeshVector=self.pyOp.getYMeshModel()
			yMeshVector=SepVector.floatVector(fromCpp=yMeshVector)
		return yMeshVector

	def getZMeshData(self):
		with pyInterpBSpline3d.ostream_redirect():
			zMeshDataVector=self.pyOp.getZMeshData()
			zMeshDataVector=SepVector.floatVector(fromCpp=zMeshDataVector)
		return zMeshDataVector

	def getXMeshData(self):
		with pyInterpBSpline3d.ostream_redirect():
			xMeshDataVector=self.pyOp.getXMeshData()
			xMeshDataVector=SepVector.floatVector(fromCpp=xMeshDataVector)
		return xMeshDataVector

	def getYMeshData(self):
		with pyInterpBSpline3d.ostream_redirect():
			yMeshDataVector=self.pyOp.getYMeshData()
			yMeshDataVector=SepVector.floatVector(fromCpp=yMeshDataVector)
		return yMeshDataVector

	def getZMeshModel1d(self):
		with pyInterpBSpline3d.ostream_redirect():
			zMesh1d=self.pyOp.getZControlPoints()
			zMesh1d=SepVector.floatVector(fromCpp=zMesh1d)
		return zMesh1d

	def getXMeshModel1d(self):
		with pyInterpBSpline3d.ostream_redirect():
			xMesh1d=self.pyOp.getXControlPoints()
			xMesh1d=SepVector.floatVector(fromCpp=xMesh1d)
		return xMesh1d

	def getYMeshModel1d(self):
		with pyInterpBSpline3d.ostream_redirect():
			yMesh1d=self.pyOp.getYControlPoints()
			yMesh1d=SepVector.floatVector(fromCpp=yMesh1d)
		return yMesh1d

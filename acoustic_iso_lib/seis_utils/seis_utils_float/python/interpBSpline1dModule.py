#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyInterpBSpline1d
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
def bSpline1dInit(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Interpolation parameters
	zOrder=parObject.getInt("zOrder",3)
	nzParam=parObject.getInt("nzParam",10000)
	scaling=parObject.getInt("scaling",1)
	zTolerance=parObject.getFloat("zTolerance",0.25)
	fat=parObject.getInt("fat",5)

	# Read data positions
	dataFile=parObject.getString("dataShape")
	dataFile=genericIO.defaultIO.getVector(dataFile)

	# z-axis
	zDataAxis=dataFile.getHyper().axes[0]
	dzData=zDataAxis.d
	nzData=zDataAxis.n
	ozData=zDataAxis.o
	fzData=ozData+(nzData-1)*dzData

	# Mesh for both directions
	zMeshFile=parObject.getString("zMesh","noZMeshFile")

	# Error tolerance
	zMeshTolerance=zTolerance*dzData

	# Compute mesh bounds (mesh should not include the fat layer)
	ozMesh=zDataAxis.o+fat*dzData
	fzMesh=zDataAxis.o+(nzData-fat-1)*dzData

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

	def getZMesh(self):
		with pyInterpBSpline1d.ostream_redirect():
			zMeshVector=self.pyOp.getZMesh()
			zMeshVector=SepVector.floatVector(fromCpp=zMeshVector)
		return zMeshVector

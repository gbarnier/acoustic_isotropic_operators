#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyInterpBSpline1dDouble
import genericIO
import SepVector
import Hypercube
import numpy as np
import math

def generateSplineMesh1d(positions,sampling,mesh):

	"""Function that creates a mesh for spline interpolation using B-splines"""

	if (mesh == "reg"):

		# Read sampling parameters
		splineMesh=[]
		oMesh=positions[0]
		fMesh=positions[len(positions)-1]
		dMesh=sampling[0]
		diff=fMesh-oMesh
		nMesh=diff/dMesh
		nMesh=math.floor(nMesh)+1

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
			# print("sampling=",sampling[iPoint])
			while (pos < positions[iPoint+1]):
				splineMesh.append(pos)
				pos=pos+sampling[iPoint]

		splineMesh.append(fMesh)

	splineMeshNp=np.asarray(splineMesh)

	return splineMeshNp

# Create an initializer
def bSpline1dDoubleInit(args):

	# Bullshit stuff
	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Interpolation parameters
	order=parObject.getInt("order",3)
	nParam=parObject.getInt("nParam")
	scaling=parObject.getInt("scaling")
	tolerance=parObject.getFloat("tolerance")
	fat=parObject.getInt("fat",5)

	# Read data positions
	dataFile=parObject.getString("vel")
	dataFile=genericIO.defaultIO.getVector(dataFile)
	dataAxis=dataFile.getHyper().axes[0]
	dData=dataAxis.d
	nData=dataAxis.n

	# Mesh
	meshFile=parObject.getString("meshFile","noMeshFile")

	# Case where user does not provides mesh
	if (meshFile=="noMeshFile"):

		# Compute mesh bounds (not including the fat)
		oMesh=dataAxis.o+fat*dData
		fMesh=dataAxis.o+(nData-fat-1)*dData

		# Get mesh parameters
		positions=parObject.getFloats("positions",[])
		positions.insert(0,oMesh)
		positions.append(fMesh)
		sampling=parObject.getFloats("sampling",[])
		mesh=parObject.getString("mesh")

		# Create mesh and convert to double1DReg
		splineMeshNp=generateSplineMesh1d(positions,sampling,mesh)
		meshAxis=Hypercube.axis(n=splineMeshNp.size)
		meshHyper=Hypercube.hypercube(axes=[meshAxis])
		splineMeshDouble=SepVector.getSepVector(meshHyper,storage="dataDouble")
		splineMeshDoubleNp=splineMeshDouble.getNdArray()
		splineMeshDoubleNp[:]=splineMeshNp

	# Case where user provides mesh
	else:

		# Read and create mesh
		splineMeshFloat=genericIO.defaultIO.getVector(meshFile)
		splineMeshFloatNp=splineMeshFloat.getNdArray()
		splineMeshDouble=SepVector.getSepVector(splineMeshFloat.getHyper(),storage="dataDouble")
		splineMeshDoubleNp=splineMeshDouble.getNdArray()
		splineMeshDoubleNp[:]=splineMeshFloatNp
		meshAxis=Hypercube.axis(n=splineMeshDoubleNp.size)
		meshHyper=Hypercube.hypercube(axes=[meshAxis])

	# Allocate model and fill with zeros
	modelDouble=SepVector.getSepVector(meshHyper,storage="dataDouble")

	# Allocate data and fill with zeros
	dataHyper=Hypercube.hypercube(axes=[dataAxis])
	dataDouble=SepVector.getSepVector(dataHyper,storage="dataDouble")

	return modelDouble,dataDouble,order,splineMeshDouble,dataAxis,nParam,scaling,tolerance,fat


class bSpline1dDouble(Op.Operator):
	def __init__(self,domain,range,order,controlPointsPosition,dataAxis,nParam,scaling,tolerance,fat):
		self.setDomainRange(domain,range)
		if("getCpp" in dir(controlPointsPosition)):
			controlPointsPosition = controlPointsPosition.getCpp()
		if("getCpp" in dir(dataAxis)):
			dataAxis = dataAxis.getCpp()
		self.pyOp = pyInterpBSpline1dDouble.interpBSpline1dDouble(order,controlPointsPosition,dataAxis,nParam,scaling,tolerance,fat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline1dDouble.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpBSpline1dDouble.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

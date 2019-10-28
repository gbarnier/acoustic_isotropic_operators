#Python module encapsulating PYBIND11 module
import pyOperator as Op
import pyInterpRbf1d
import genericIO
import SepVector
import Hypercube
import numpy as np

def generateSplineMesh1d(positions,sampling,mesh,tolerance):

	"""Function that creates a mesh for RBF interpolation"""

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

def interpRbf1dInit(args):

	# IO object
	parObject=genericIO.io(params=args)

	epsilon=parObject.getFloat("epsilon")
	fat=parObject.getInt("fat",5)
	scaling=parObject.getInt("scaling",1)
	zTolerance=parObject.getFloat("zTolerance",0.25)

	# Read data dimensions
	dataFile=parObject.getString("dataShape")
	dataFile=genericIO.defaultIO.getVector(dataFile)
	zDataAxis=dataFile.getHyper().axes[0]
	dzData=zDataAxis.d
	nzData=zDataAxis.n
	ozData=zDataAxis.o
	fzData=ozData+(nzData-1)*dzData

	# Mesh for both directions
	zMeshFile=parObject.getString("zMesh","noZMeshFile")

	# Compute mesh bounds (mesh should not include the fat layer)
	ozMesh=zDataAxis.o+fat*dzData
	fzMesh=zDataAxis.o+(nzData-fat-1)*dzData

	# Error tolerance
	zMeshTolerance=zTolerance*dzData

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

	return model,data,epsilon,zSplineMesh,zDataAxis,scaling,fat

class interpRbf1d(Op.Operator):

	def __init__(self,domain,range,epsilon,zModel,zDataAxis,scaling,fat):

		self.setDomainRange(domain,range)
		if("getCpp" in dir(zModel)):
			zModel = zModel.getCpp()
		if("getCpp" in dir(zDataAxis)):
			zDataAxis = zDataAxis.getCpp()
		self.pyOp = pyInterpRbf1d.interpRbf1d(epsilon,zModel,zDataAxis,scaling,fat)
		return

	def forward(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpRbf1d.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyInterpRbf1d.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def getZMesh(self):
		with pyInterpRbf1d.ostream_redirect():
			zMeshVector=self.pyOp.getZMesh()
			zMeshVector=SepVector.floatVector(fromCpp=zMeshVector)
		return zMeshVector

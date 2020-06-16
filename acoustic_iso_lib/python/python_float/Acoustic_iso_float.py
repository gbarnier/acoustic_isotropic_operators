# Python module encapsulating PYBIND11 module
# It seems necessary to allow std::cout redirection to screen
import pyAcoustic_iso_float_nl
import pyAcoustic_iso_float_born
import pyAcoustic_iso_float_born_ext
import pyAcoustic_iso_float_tomo
import pyAcoustic_iso_float_wemva
import pyOperator as Op
import spatialDerivModule
import timeIntegModule
import dataTaperModule

# Other necessary modules
import genericIO
import SepVector
import Hypercube
import numpy as np

from pyAcoustic_iso_float_nl import deviceGpu


############################ Dask interface ####################################
#Dask-related modules and functions
import dask.distributed as daskD
from dask_util import DaskClient
import pyDaskVector
import re

def create_client(parObject):
	"""
	   Function to create Dask client if requested
	"""
	hostnames = parObject.getString("hostnames","noHost")
	pbs_args = parObject.getString("pbs_args","noPBS")
	lsf_args = parObject.getString("lsf_args","noLSF")
	cluster_args = None
	if pbs_args != "noPBS":
		cluster_args = pbs_args
		cluster_name = "pbs_params"
	elif lsf_args != "noLSF":
		cluster_args = lsf_args
		cluster_name = "lsf_params"
	if hostnames != "noHost" and cluster_args is not None:
		raise ValueError("Only one interface can be used for a client! User provided both SSH and PBS/LSF parameters!")
	#Starting Dask client if requested
	client = None
	nWrks = None
	args = None
	if hostnames != "noHost":
		args = {"hostnames":hostnames.split(",")}
		scheduler_file = parObject.getString("scheduler_file","noFile")
		if scheduler_file != "noFile":
			args.update({"scheduler_file_prefix":scheduler_file})
		print("Starting Dask client using the following workers: %s"%(hostnames))
	elif cluster_args:
		n_wrks = parObject.getInt("n_wrks",1)
		n_jobs = parObject.getInt("n_jobs")
		args = {"n_jobs":n_jobs}
		args.update({"n_wrks":n_wrks})
		cluster_dict={elem.split(";")[0] : elem.split(";")[1] for elem in cluster_args.split(",")}
		if "cores" in cluster_dict.keys():
			cluster_dict.update({"cores":int(cluster_dict["cores"])})
		if "mem" in cluster_dict.keys():
			cluster_dict.update({"mem":int(cluster_dict["mem"])})
		if "ncpus" in cluster_dict.keys():
			cluster_dict.update({"ncpus":int(cluster_dict["ncpus"])})
		if "nanny" in cluster_dict.keys():
			nanny_flag = True
			if cluster_dict["nanny"] in "0falseFalse":
				nanny_flag = False
			cluster_dict.update({"nanny":nanny_flag})
		if "dashboard_address" in cluster_dict.keys():
			if cluster_dict["dashboard_address"] in "Nonenone":
				cluster_dict.update({"dashboard_address":None})
		if "env_extra" in cluster_dict.keys():
			cluster_dict.update({"env_extra":cluster_dict["env_extra"].split(":")})
		if "job_extra" in cluster_dict.keys():
			cluster_dict.update({"job_extra":cluster_dict["job_extra"].split("|")})
		cluster_dict={cluster_name:cluster_dict}
		args.update(cluster_dict)
		print("Starting jobqueue Dask client using %s workers on %s jobs"%(n_wrks,n_jobs))

	if args:
		client = DaskClient(**args)
		print("Client has started!")
		nWrks = client.getNworkers()
	return client, nWrks

def parfile2pars(args):
	"""Function to expand arguments in parfile to parameters"""
	#Check if par argument was provided
	par_arg = None
	match = [arg for arg in args if "par" in arg]
	if len(match) > 0:
		par_arg = match[-1] #Taking last par argument
	#If par was found expand arguments
	if par_arg:
		par_file = par_arg.split("=")[-1]
		with open(par_file) as fid:
			lines = fid.read().splitlines()
		#Substitute par with its arguments
		idx = args.index(par_arg)
		args = args[:idx] + lines + args[idx+1:]
	return args

def create_parObj(args):
	"""Function to call genericIO correctly"""
	obj = genericIO.io(params=args)
	return obj

def spreadParObj(client,args,par):
	"""Function to spread parameter object to workers"""
	#Spreading/Instantiating the parameter objects
	List_Shots = par.getInts("nShot",0)
	parObject = []
	args1=parfile2pars(args)
	#Finding index of nShot parameter
	idx_nshot = [ii for ii,el in enumerate(args1) if "nShot" in el][-1]
	for idx,wrkId in enumerate(client.getWorkerIds()):
		#Substituting nShot with the correct number of shots
		args1[idx_nshot]="nShot=%s"%(List_Shots[idx])
		parObject.append(client.getClient().submit(create_parObj,args1,workers=[wrkId],pure=False))
	daskD.wait(parObject)
	return parObject

def call_deviceGpu(nzDevice, ozDevice, dzDevice, nxDevice, oxDevice, dxDevice, vel, nts, dipole, zDipoleShift, xDipoleShift):
	"""Function instantiate a deviceGpu object (first constructor)"""
	obj = deviceGpu(nzDevice, ozDevice, dzDevice, nxDevice, oxDevice, dxDevice, vel.getCpp(), nts, dipole, zDipoleShift, xDipoleShift)
	return obj

def get_axes(vecObj):
	"""Function to return Axes from vector"""
	return vecObj.getHyper().axes

def call_deviceGpu1(z_dev, x_dev, vel, nts, dipole, zDipoleShift, xDipoleShift):
	"""Function to construct device using its absolute position"""
	zCoordFloat=SepVector.getSepVector(ns=[1])
	xCoordFloat=SepVector.getSepVector(ns=[1])
	zCoordFloat.set(z_dev)
	xCoordFloat.set(x_dev)
	obj = deviceGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(), vel.getCpp(), nts, dipole, zDipoleShift, xDipoleShift)
	return obj

def chunkData(dataVecLocal,dataSpaceRemote):
	"""Function to chunk and spread the data vector across dask workers"""
	dask_client = dataSpaceRemote.dask_client #Getting Dask client
	client = dask_client.getClient()
	wrkIds = dask_client.getWorkerIds()
	dataAxes = client.gather(client.map(get_axes,dataSpaceRemote.vecDask,pure=False)) #Getting hypercubes of remote vector chunks
	List_Shots = [axes[-1].n for axes in dataAxes]
	dataNd = dataVecLocal.getNdArray()
	if(np.sum(List_Shots) != dataNd.shape[0]):
		raise ValueError("Number of shot within provide data vector (%s) not consistent with total number of shots from nShot parameter (%s)"%(dataNd.shape[0],np.sum(List_Shots)))
	#Pointer-wise chunking
	dataArrays = []
	firstShot = 0
	for nShot in List_Shots:
		dataArrays.append(dataNd[firstShot:firstShot+nShot,:,:])
		firstShot += nShot
	#Copying the data to remove vector
	dataVecRemote = dataSpaceRemote.clone()
	for idx,wrkId in enumerate(wrkIds):
		arrD = client.scatter(dataArrays[idx],workers=[wrkId])
		daskD.wait(arrD)
		daskD.wait(client.submit(pyDaskVector.copy_from_NdArray,dataVecRemote.vecDask[idx],arrD,pure=False))
	# daskD.wait(client.map(pyDaskVector.copy_from_NdArray,dataVecRemote.vecDask,dataArrays,pure=False))
	return dataVecRemote

############################ Bounds vectors ####################################
# Create bound vectors for FWI
def createBoundVectors(parObject,model):

	# Get model dimensions
	nz=parObject.getInt("nz")
	nx=parObject.getInt("nx")
	fat=parObject.getInt("fat")
	spline=parObject.getInt("spline",0)
	if (spline==1): fat=0

	# Min bound
	minBoundVectorFile=parObject.getString("minBoundVector","noMinBoundVectorFile")
	if (minBoundVectorFile=="noMinBoundVectorFile"):
		minBound=parObject.getFloat("minBound")
		minBoundVector=model.clone()
		minBoundVector.scale(0.0)
		minBoundVectorNd=minBoundVector.getNdArray()
		for ix in range(fat,nx-fat):
			for iz in range(fat,nz-fat):
				minBoundVectorNd[ix][iz]=minBound

	else:
		minBoundVector=genericIO.defaultIO.getVector(minBoundVectorFile)

	# Max bound
	maxBoundVectorFile=parObject.getString("maxBoundVector","noMaxBoundVectorFile")
	if (maxBoundVectorFile=="noMaxBoundVectorFile"):
		maxBound=parObject.getFloat("maxBound")
		maxBoundVector=model.clone()
		maxBoundVector.scale(0.0)
		maxBoundVectorNd=maxBoundVector.getNdArray()
		for ix in range(fat,nx-fat):
			for iz in range(fat,nz-fat):
				maxBoundVectorNd[ix][iz]=maxBound

	else:
		maxBoundVector=genericIO.defaultIO.getVector(maxBoundVectorFile)


	return minBoundVector,maxBoundVector

############################ Acquisition geometry ##############################
# Build sources geometry
def buildSourceGeometry(parObject,vel,client=None):

	#Common parameters
	sourceGeomFile = parObject.getString("sourceGeomFile","None")
	nShot = parObject.getInt("nShot")
	nts = parObject.getInt("nts")
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getInt("zDipoleShift",2)
	xDipoleShift = parObject.getInt("xDipoleShift",0)
	sourcesVector=[]

	#Reading source geometry from file
	if(sourceGeomFile != "None"):
		sourceGeomVectorNd = genericIO.defaultIO.getVector(sourceGeomFile).getNdArray()
		zCoordFloat=SepVector.getSepVector(ns=[1])
		xCoordFloat=SepVector.getSepVector(ns=[1])
		#Check for consistency between number of shots and provided coordinates
		if(nShot != sourceGeomVectorNd.shape[1]):
			raise ValueError("Number of shots (#shot=%s) not consistent with geometry file (#shots=%s)!"%(nShot,sourceGeomVectorNd.shape[1]))
		#Setting source geometry
		for ishot in range(nShot):
			#Setting z and x position of the source for the given experiment
			zCoordFloat.set(sourceGeomVectorNd[2,ishot])
			xCoordFloat.set(sourceGeomVectorNd[0,ishot])
			sourcesVector.append(deviceGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(), vel.getCpp(), nts, dipole, zDipoleShift, xDipoleShift))
		sourceAxis=Hypercube.axis(n=nShot,o=1.0,d=1.0)

	#Reading regular source geometry from parameters
	else:
		# Horizontal axis
		dx=vel.getHyper().axes[1].d
		ox=vel.getHyper().axes[1].o
		# Sources geometry
		nzSource=1
		dzSource=1
		nxSource=1
		dxSource=1
		ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
		oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
		spacingShots=parObject.getInt("spacingShots")
		sourceAxis=Hypercube.axis(n=nShot,o=ox+oxSource*dx,d=spacingShots*dx)
		#Setting source geometry
		for ishot in range(nShot):
			sourcesVector.append(deviceGpu(nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, vel.getCpp(), nts, dipole, zDipoleShift, xDipoleShift))
			oxSource=oxSource+spacingShots # Shift source

	return sourcesVector,sourceAxis

# Build sources geometry for Dask
def buildSourceGeometryDask(parObject,vel,hyper_vel,client):

	#Common parameters
	sourceGeomFile = parObject.getString("sourceGeomFile","None")
	List_Shots = parObject.getInts("nShot",0)
	nts = parObject.getInt("nts")
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getInt("zDipoleShift",2)
	xDipoleShift = parObject.getInt("xDipoleShift",0)

	#Checking if list of shots is consistent with number of workers
	nWrks = client.getNworkers()
	wrkIds = client.getWorkerIds()
	if len(List_Shots) != nWrks:
		raise ValueError("Number of workers (#nWrk=%s) not consistent with length of the provided list of shots (nShot=%s)"%(nWrks,parObject.getString("nShot")))

	sourcesVector = [[] for ii in range(nWrks)]
	sourceAxis = []

	#Reading source geometry from file
	if(sourceGeomFile != "None"):
		sourceGeomVectorNd = genericIO.defaultIO.getVector(sourceGeomFile).getNdArray()
		for idx,nShot in enumerate(List_Shots):
			#Check for consistency between number of shots and provided coordinates
			if(nShot != sourceGeomVectorNd.shape[1]):
				raise ValueError("Number of shots (#shot=%s) not consistent with geometry file (#shots=%s)!"%(nShot,sourceGeomVectorNd.shape[1]))
			#Setting source geometry
			for ishot in range(nShot):
				sourcesVector[idx].append(client.getClient().submit(call_deviceGpu1, sourceGeomVectorNd[2,ishot],sourceGeomVectorNd[0,ishot], vel[idx], nts, dipole, zDipoleShift, xDipoleShift,workers=wrkIds[idx],pure=False))
			daskD.wait(sourcesVector[idx])
			sourceAxis.append(Hypercube.axis(n=nShot,o=1.0,d=1.0))

	#Reading regular source geometry from parameters
	else:
		# Horizontal axis
		dx=hyper_vel.axes[1].d
		ox=hyper_vel.axes[1].o
		# Sources geometry
		nzSource=1
		dzSource=1
		nxSource=1
		dxSource=1
		ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
		oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
		spacingShots=parObject.getInt("spacingShots")
		#Source position and sampling
		ox = ox+oxSource*dx
		dx = spacingShots*dx
		for idx,nShot in enumerate(List_Shots):
			#Shot axis for given shots
			sourceAxis.append(Hypercube.axis(n=nShot,o=ox,d=dx))
			#Setting source geometry
			for ishot in range(nShot):
				sourcesVector[idx].append(client.getClient().submit(call_deviceGpu, nzSource, ozSource, dzSource, nxSource, oxSource, dxSource, vel[idx], nts, dipole, zDipoleShift, xDipoleShift,workers=wrkIds[idx],pure=False))
				oxSource+=spacingShots # Shift source
			daskD.wait(sourcesVector[idx])
			#Adding shots offset to origin of shot axis
			ox += (nShot-1)*dx

	return sourcesVector,sourceAxis

# Build sources geometry for dipole only
def buildSourceGeometryDipole(parObject,vel):

	# Horizontal axis
	dx=vel.getHyper().axes[1].d
	ox=vel.getHyper().axes[1].o

	# Sources geometry
	nzSource=1
	ozSource=parObject.getInt("zSource")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")

	# Shift the source depth shallower to account for the Dz in Symes' formula and so that
	# the resulting spatial derivative is on the same grid
	ozSource=ozSource-parObject.getInt("SymesDzHalfStencil",1)

	dzSource=1
	nxSource=1
	oxSource=parObject.getInt("xSource")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
	dxSource=1
	spacingShots=parObject.getInt("spacingShots")
	sourceAxis=Hypercube.axis(n=parObject.getInt("nShot"),o=ox+oxSource*dx,d=spacingShots*dx)
	sourcesVector=[]

	# Modify the dipole shift for the z-derivative in Symes' pseudo-inverse
	zDipoleShift=2*parObject.getInt("SymesDzHalfStencil",1)
	if (parObject.getInt("dipole",0)==0):
		ozSource=ozSource-parObject.getInt("SymesDzHalfStencil",1)

	for ishot in range(parObject.getInt("nShot")):
		sourcesVector.append(deviceGpu(nzSource,ozSource,dzSource,nxSource,oxSource,dxSource,vel.getCpp(),parObject.getInt("nts"),1,zDipoleShift, parObject.getInt("xDipoleShift",0)))
		oxSource=oxSource+spacingShots # Shift source

	return sourcesVector,sourceAxis

# Build receivers geometry
def buildReceiversGeometry(parObject,vel,client=None):

	# Two possible cases:

	# I. Irregular receivers' geometry. The user needs to provide a receiver geometry file which is a 3D array.
	# First (fast) axis contains the coordinates (x,y,z)
	# Second axis contains the receivers index
	# Third axis contains the shots index

	# II. Regular and constant receivers' geometry. The user needs to provide the receivers' positions in a grid format (o,d,n)

	nts = parObject.getInt("nts")
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getInt("zDipoleShift",2)
	xDipoleShift = parObject.getInt("xDipoleShift",0)
	receiverGeomFile = parObject.getString("receiverGeomFile","None")
	receiversVector=[]

	if (receiverGeomFile != "None"):

		# Read geometry file: 3 axes
		# First (fast) axis: spatial coordinates
		# Second axis: receiver index !!! The number of receivers per shot must be constant
		# Third axis: shot index

		receiverGeomVectorNd = genericIO.defaultIO.getVector(receiverGeomFile).getNdArray()

		# Check consistency with total number of shots
		nShot = parObject.getInt("nShot",-1)
		if (nShot != receiverGeomVectorNd.shape[2]):
				raise ValueError("**** ERROR [buildReceiversGeometry]: Number of shots from parfile (#shot=%s) not consistent with receivers' geometry file (#shots=%s) ****\n"%(nShot,receiverGeomVectorNd.shape[2]))

		# Read size of receivers' geometry file
		nReceiverPerShot = parObject.getInt("nReceiverPerShot",-1) # -> might move this call to the irregular geometry case
		if(nReceiverPerShot != receiverGeomVectorNd.shape[1]):
				raise ValueError("**** ERROR [buildReceiversGeometry]: Number of receivers from parfile (#nReceiverPerShot=%s) not consistent with receivers' geometry file (#recs=%s) ****\n"%(nReceiverPerShot,receiverGeomVectorNd.shape[1]))

		# Create inputs for deviceGpu constructor
		zCoordFloat=SepVector.getSepVector(ns=[nReceiverPerShot])
		xCoordFloat=SepVector.getSepVector(ns=[nReceiverPerShot])

		# Generate vector containing deviceGpu objects
		for ishot in range(nShot):

				# Update the receiver's coordinates
				zCoordFloat.set(receiverGeomVectorNd[2,:,ishot])
				xCoordFloat.set(receiverGeomVectorNd[0,:,ishot])
				receiversVector.append(deviceGpu(zCoordFloat.getCpp(), xCoordFloat.getCpp(), vel.getCpp(), nts, dipole, zDipoleShift, xDipoleShift))

		# Generate receiver axis
		receiverAxis=Hypercube.axis(n=nReceiverPerShot,o=0.0,d=1.0)

	else:

		#Horizontal axis information
		dx=vel.getHyper().axes[1].d
		ox=vel.getHyper().axes[1].o

		# Generate only one deviceGpu object with all the receivers' coordinates
		nzReceiver=1
		ozReceiver=parObject.getInt("depthReceiver")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
		dzReceiver=1
		nxReceiver=parObject.getInt("nReceiver")
		oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
		dxReceiver=parObject.getInt("dReceiver")
		receiverAxis=Hypercube.axis(n=nxReceiver,o=ox+oxReceiver*dx,d=dxReceiver*dx)

		nRecGeom=1; # Constant receivers' geometry
		for iRec in range(nRecGeom):
				receiversVector.append(deviceGpu(nzReceiver,ozReceiver,dzReceiver,nxReceiver,oxReceiver,dxReceiver,vel.getCpp(),nts, dipole, zDipoleShift, xDipoleShift))

	return receiversVector,receiverAxis

def buildReceiversGeometryDask(parObject,vel,hyper_vel,client=None):

	# Horizontal axis
	dx=hyper_vel.axes[1].d
	ox=hyper_vel.axes[1].o
	nts = parObject.getInt("nts")
	dipole = parObject.getInt("dipole",0)
	zDipoleShift = parObject.getInt("zDipoleShift",2)
	xDipoleShift = parObject.getInt("xDipoleShift",0)

	nzReceiver=1
	ozReceiver=parObject.getInt("depthReceiver")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")
	dzReceiver=1
	nxReceiver=parObject.getInt("nReceiver")
	oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
	dxReceiver=parObject.getInt("dReceiver")

	#Getting number of workers
	nWrks = client.getNworkers()
	wrkIds = client.getWorkerIds()

	receiverAxis=[Hypercube.axis(n=nxReceiver,o=ox+oxReceiver*dx,d=dxReceiver*dx)]*nWrks
	receiversVector = [[] for ii in range(nWrks)]
	for iwrk in range(nWrks):
		receiversVector[iwrk].append(client.getClient().submit(call_deviceGpu,nzReceiver,ozReceiver,dzReceiver,nxReceiver,oxReceiver,dxReceiver,vel[iwrk],nts, dipole, zDipoleShift, xDipoleShift,workers=wrkIds[iwrk],pure=False))
	daskD.wait(receiversVector)

	return receiversVector,receiverAxis

# Build receivers geometry for dipole only
def buildReceiversGeometryDipole(parObject,vel):

	# Horizontal axis
	dx=vel.getHyper().axes[1].n
	dx=vel.getHyper().axes[1].d
	ox=vel.getHyper().axes[1].o

	nzReceiver=1
	ozReceiver=parObject.getInt("depthReceiver")-1+parObject.getInt("zPadMinus")+parObject.getInt("fat")

	# Shift the source depth shallower to account for the Dz in Symes' formula and so that
	# the resulting spatial derivative is on the same grid
	if (parObject.getInt("dipole",0)==0):
		ozReceiver=ozReceiver-parObject.getInt("SymesDzHalfStencil",1)

	dzReceiver=1
	nxReceiver=parObject.getInt("nReceiver")
	oxReceiver=parObject.getInt("oReceiver")-1+parObject.getInt("xPadMinus")+parObject.getInt("fat")
	dxReceiver=parObject.getInt("dReceiver")
	receiverAxis=Hypercube.axis(n=nxReceiver,o=ox+oxReceiver*dx,d=dxReceiver*dx)
	receiversVector=[]
	nRecGeom=1; # Constant receivers' geometry

	# Modify the dipole shift for the z-derivative in Symes' pseudo-inverse
	zDipoleShift=2*parObject.getInt("SymesDzHalfStencil",1)
	for iRec in range(nRecGeom):
		receiversVector.append(deviceGpu(nzReceiver,ozReceiver,dzReceiver,nxReceiver,oxReceiver,dxReceiver,vel.getCpp(),parObject.getInt("nts"),1,zDipoleShift,parObject.getInt("xDipoleShift",0)))

	return receiversVector,receiverAxis

############################### Nonlinear ######################################
def nonlinearOpInitFloat(args,client=None):
	"""Function to correctly initialize nonlinear operator
	   The function will return the necessary variables for operator construction
	"""
	# IO object
	parObject=genericIO.io(params=args)

	# Allocate and read velocity
	velFile=parObject.getString("vel","noVelFile")
	if (velFile == "noVelFile"):
		print("**** ERROR: User did not provide velocity file ****\n")
		quit()
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate model and fill with zeros
	dummyAxis=Hypercube.axis(n=1)
	modelHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis,dummyAxis])
	modelFloat=SepVector.getSepVector(modelHyper)

	#Local vector copy useful for dask interface
	modelFloatLocal = modelFloat

	#Setting variables if Dask is employed
	if client:
		#Getting number of workers and passing
		nWrks = client.getNworkers()
		#Spreading domain vector (i.e., wavelet)
		modelFloat = pyDaskVector.DaskVector(client,vectors=[modelFloat]*nWrks)

		#Spreading velocity model to workers
		hyper_vel = velFloat.getHyper()
		velFloat = pyDaskVector.DaskVector(client,vectors=[velFloat]*nWrks).vecDask

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometryDask(parObject,velFloat,hyper_vel,client)
		receiversVector,receiverAxis=buildReceiversGeometryDask(parObject,velFloat,hyper_vel,client)

		#Allocating data vectors to be spread
		dataFloat = []
		for iwrk in range(nWrks):
			dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis[iwrk],sourceAxis[iwrk]])
			dataFloat.append(SepVector.getSepVector(dataHyper))
		dataFloat = pyDaskVector.DaskVector(client,vectors=dataFloat,copy=False)

		#Spreading/Instantiating the parameter objects
		parObject = spreadParObj(client,args,parObject)

	else:

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
		receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

		# Allocate data and fill with zeros
		dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis])
		dataFloat=SepVector.getSepVector(dataHyper)

	# Outputs
	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,receiversVector,modelFloatLocal

class nonlinearPropShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear propagator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,receiversVector):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		self.pyOp = pyAcoustic_iso_float_nl.nonlinearPropShotsGpu(velocity,paramP.param,sourceVector,receiversVector)
		return

	def __str__(self):
		"""Name of the operator"""
		return " NLOper "

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_nl.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

	def getWfld(self):
		with pyAcoustic_iso_float_nl.ostream_redirect():
			wfld = self.pyOp.getWfld()
			wfld = SepVector.floatVector(fromCpp=wfld)
		return wfld
def nonlinearFwiOpInitFloat(args,client=None):
	"""Function to correctly initialize a nonlinear operator where the model is velocity
	   The function will return the necessary variables for operator construction
	"""
	# IO object
	parObject=genericIO.io(params=args)

	# Allocate and read starting model
	modelStartFile=parObject.getString("vel")
	modelStart=genericIO.defaultIO.getVector(modelStartFile)

	#Local vector copy useful for dask interface
	modelStartLocal = modelStart

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Allocate wavelet and fill with zeros
	dummyAxis=Hypercube.axis(n=1)
	sourcesSignalHyper=Hypercube.hypercube(axes=[timeAxis,dummyAxis,dummyAxis])
	sourcesSignal=SepVector.getSepVector(sourcesSignalHyper)

	#Setting variables if Dask is employed
	if client:

		#Getting number of workers and passing
		nWrks = client.getNworkers()

		#Spreading source wavelet
		sourcesSignal = pyDaskVector.DaskVector(client,vectors=[sourcesSignal]*nWrks)

		#Spreading domain vector (i.e., velocity)
		hyper_model = modelStart.getHyper()
		modelStart = pyDaskVector.DaskVector(client,vectors=[modelStart]*nWrks)

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometryDask(parObject,modelStart.vecDask,hyper_model,client)
		receiversVector,receiverAxis=buildReceiversGeometryDask(parObject,modelStart.vecDask,hyper_model,client)

		#Allocating data vectors to be spread
		dataFloat = []
		for iwrk in range(nWrks):
			dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis[iwrk],sourceAxis[iwrk]])
			dataFloat.append(SepVector.getSepVector(dataHyper))
		dataFloat = pyDaskVector.DaskVector(client,vectors=dataFloat,copy=False)

		#Spreading/Instantiating the parameter objects
		parObject = spreadParObj(client,args,parObject)

	else:


		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometry(parObject,modelStart)
		receiversVector,receiverAxis=buildReceiversGeometry(parObject,modelStart)

		# Allocate data and fill with zeros
		dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis])
		dataFloat=SepVector.getSepVector(dataHyper)

	# Outputs
	return modelStart,dataFloat,sourcesSignal,parObject,sourcesVector,receiversVector,modelStartLocal

class nonlinearFwiPropShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for non-linear propagator where the model vector is the velocity"""

	def __init__(self,domain,range,sources,paramP,sourceVector,receiversVector):
		#Domain = velocity model
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(sources)):
			sources = sources.getCpp()
			self.sources = sources.clone()
		self.pyOp = pyAcoustic_iso_float_nl.nonlinearPropShotsGpu(domain,paramP.param,sourceVector,receiversVector)
		return

	def __str__(self):
		"""Name of the operator"""
		return " NLOper "

	def forward(self,add,model,data):
		#Setting velocity model
		self.setVel(model)
		#Checking if getCpp is present
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.forward(add,self.sources,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_nl.ostream_redirect():
			self.pyOp.setVel(vel)
		return

################################### Born #######################################
def BornOpInitFloat(args,client=None):
	"""Function to correctly initialize Born operator
	   The function will return the necessary variables for operator construction
	"""

	# IO object
	parObject=genericIO.io(params=args)

	# Velocity model
	velFile=parObject.getString("vel")
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Allocate model
	modelFloat=SepVector.getSepVector(velFloat.getHyper())
	#Local vector copy useful for dask interface
	modelFloatLocal = modelFloat

	# Time Axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Read sources signals
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		raise IOError("**** ERROR: User did not provide seismic sources file ****\n")

	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)

	if client:
		#Getting number of workers and passing
		nWrks = client.getNworkers()

		#Spreading velocity model to workers
		hyper_vel = velFloat.getHyper()
		velFloatD = pyDaskVector.DaskVector(client,vectors=[velFloat]*nWrks)
		velFloat = velFloatD.vecDask

		#Allocate model
		modelFloat = velFloatD.clone()
		modelFloat.zero()

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometryDask(parObject,velFloat,hyper_vel,client)
		receiversVector,receiverAxis=buildReceiversGeometryDask(parObject,velFloat,hyper_vel,client)

		#Spreading source wavelet
		sourcesSignalsFloat = pyDaskVector.DaskVector(client,vectors=[sourcesSignalsFloat]*nWrks).vecDask
		sourcesSignalsVector = client.getClient().map((lambda x: [x]),sourcesSignalsFloat,pure=False)
		daskD.wait(sourcesSignalsVector)

		#Allocating data vectors to be spread
		dataFloat = []
		for iwrk in range(nWrks):
			dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis[iwrk],sourceAxis[iwrk]])
			dataFloat.append(SepVector.getSepVector(dataHyper))
		dataFloat = pyDaskVector.DaskVector(client,vectors=dataFloat,copy=False)

		#Spreading/Instantiating the parameter objects
		parObject = spreadParObj(client,args,parObject)

	else:

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
		receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

		sourcesSignalsVector=[]
		sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

		# Allocate data
		dataFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis]))

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,modelFloatLocal

class BornShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		self.pyOp = pyAcoustic_iso_float_born.BornShotsGpu(velocity,paramP.param,sourceVector,sourcesSignalsVector,receiversVector)
		return

	def __str__(self):
		"""Name of the operator"""
		return " BornOp "

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_born.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_born.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

	def getSrcWfld(self):
		with pyAcoustic_iso_float_born.ostream_redirect():
			wfld = self.pyOp.getSrcWfld()
			wfld = SepVector.floatVector(fromCpp=wfld)
		return wfld

	def getSecWfld(self):
		with pyAcoustic_iso_float_born.ostream_redirect():
			wfld = self.pyOp.getSecWfld()
			wfld = SepVector.floatVector(fromCpp=wfld)
		return wfld
############################## Born extended ###################################
def BornExtOpInitFloat(args,client=None):

	# IO object
	parObject=genericIO.io(params=args)

	# Velocity model
	velFile=parObject.getString("vel", "noVelFile")
	if (velFile == "noVelFile"):
		print("**** ERROR: User did not provide vel file ****\n")
		quit()
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Space axes
	zAxis=velFloat.getHyper().axes[0]
	xAxis=velFloat.getHyper().axes[1]

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Extension axis
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR: User did not provide extension type ****\n")
		quit()

	nExt=parObject.getInt("nExt", -1)
	if (nExt == -1):
		print("**** ERROR: User did not provide size of extension axis ****\n")
		quit()
	if (nExt%2 ==0):
		print("Length of extension axis must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt=parObject.getFloat("dts",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt=parObject.getFloat("dx",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	extAxis=Hypercube.axis(n=nExt,o=oExt,d=dExt) # Create extended axis

	# Read sources signals (we assume one unique point source signature for all shots)
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)

	# Allocate model
	modelFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,extAxis]))
	#Local vector copy useful for dask interface
	modelFloatLocal = modelFloat

	if client:

		#Getting number of workers and passing
		nWrks = client.getNworkers()

		#Spreading velocity model to workers
		hyper_vel = velFloat.getHyper()
		velFloatD = pyDaskVector.DaskVector(client,vectors=[velFloat]*nWrks)
		velFloat = velFloatD.vecDask

		modelFloat = pyDaskVector.DaskVector(client,vectors=[modelFloat]*nWrks)

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometryDask(parObject,velFloat,hyper_vel,client)
		receiversVector,receiverAxis=buildReceiversGeometryDask(parObject,velFloat,hyper_vel,client)

		#Spreading source wavelet
		sourcesSignalsFloat = pyDaskVector.DaskVector(client,vectors=[sourcesSignalsFloat]*nWrks).vecDask
		sourcesSignalsVector = client.getClient().map((lambda x: [x]),sourcesSignalsFloat,pure=False)
		daskD.wait(sourcesSignalsVector)

		#Allocating data vectors to be spread
		dataFloat = []
		for iwrk in range(nWrks):
			dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis[iwrk],sourceAxis[iwrk]])
			dataFloat.append(SepVector.getSepVector(dataHyper))
		dataFloat = pyDaskVector.DaskVector(client,vectors=dataFloat,copy=False)

		#Spreading/Instantiating the parameter objects
		parObject = spreadParObj(client,args,parObject)

	else:

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
		receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

		sourcesSignalsVector=[]
		sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

		# Allocate data
		dataFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis]))

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,modelFloatLocal

class BornExtShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Extended Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		self.pyOp = pyAcoustic_iso_float_born_ext.BornExtShotsGpu(velocity,paramP.param,sourceVector,sourcesSignalsVector,receiversVector)
		return

	def __str__(self):
		"""Name of the operator"""
		return " BornExt"

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def add_spline(self,Spline_op):
		"""
		   Adding spline operator to set background
		"""
		self.Spline_op = Spline_op
		self.tmp_fine_model = Spline_op.range.clone()
		return

	def setVel(self,vel_in):
		if("Spline_op" in dir(self)):
			self.Spline_op.forward(False,vel_in,self.tmp_fine_model)
			vel = self.tmp_fine_model
		else:
			vel = vel_in
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def getVel(self):
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			vel = self.pyOp.getVel()
			vel = SepVector.floatVector(fromCpp=vel)
		return vel

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

############################## Tomo nonlinear #################################
def BornExtTomoInvOpInitFloat(args,client=None):

	# IO object
	parObject=genericIO.io(params=args)

	# Velocity model
	modelStartFile=parObject.getString("vel")
	modelStart=genericIO.defaultIO.getVector(modelStartFile)

	#Local vector copy useful for dask interface
	modelStartLocal = modelStart

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Extended reflectivity
	reflectivityFile=parObject.getString("reflectivity")
	reflectivityFloat=genericIO.defaultIO.getVector(reflectivityFile,ndims=3)

	# Read sources signals (we assume one unique point source signature for all shots)
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)

	if client:
		#Getting number of workers and passing
		nWrks = client.getNworkers()

		#Spreading velocity model to workers
		hyper_vel = modelStart.getHyper()
		velFloatD = pyDaskVector.DaskVector(client,vectors=[modelStart]*nWrks)
		velFloat = velFloatD.vecDask

		modelStart = pyDaskVector.DaskVector(client,vectors=[modelStart]*nWrks)

		#Spreading reflectivity
		reflectivityFloat = pyDaskVector.DaskVector(client,vectors=[reflectivityFloat]*nWrks).vecDask

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometryDask(parObject,velFloat,hyper_vel,client)
		receiversVector,receiverAxis=buildReceiversGeometryDask(parObject,velFloat,hyper_vel,client)

		#Spreading source wavelet
		sourcesSignalsFloat = pyDaskVector.DaskVector(client,vectors=[sourcesSignalsFloat]*nWrks).vecDask
		sourcesSignalsVector = client.getClient().map((lambda x: [x]),sourcesSignalsFloat,pure=False)
		daskD.wait(sourcesSignalsVector)

		#Allocating data vectors to be spread
		dataFloat = []
		for iwrk in range(nWrks):
			dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis[iwrk],sourceAxis[iwrk]])
			dataFloat.append(SepVector.getSepVector(dataHyper))
		dataFloat = pyDaskVector.DaskVector(client,vectors=dataFloat,copy=False)

		#Spreading/Instantiating the parameter objects
		parObject = spreadParObj(client,args,parObject)

	else:
		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometry(parObject,modelStart)
		receiversVector,receiverAxis=buildReceiversGeometry(parObject,modelStart)

		sourcesSignalsVector=[]
		sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

		# Allocate data
		dataFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis]))

	return modelStart,dataFloat,reflectivityFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,modelStartLocal

class BornExtTomoInvShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Extended Born operator"""

	def __init__(self,domain,range,reflectivity,paramP,sourceVector,sourcesSignalsVector,receiversVector):
		# Domain = velocity model
		# Range = recorded data space
		self.setDomainRange(domain,range)
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(reflectivity)):
			reflectivity = reflectivity.getCpp()
			self.reflectivity = reflectivity.clone()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		self.pyOp = pyAcoustic_iso_float_born_ext.BornExtShotsGpu(domain,paramP.param,sourceVector,sourcesSignalsVector,receiversVector)
		return

	def __str__(self):
		"""Name of the operator"""
		return " BornExt"

	def forward(self,add,model,data):
		self.setVel(model)
		# Checking if getCpp is present
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.forward(add,self.reflectivity,data)
		return

	def add_spline(self,Spline_op):
		"""
		   Adding spline operator to set background
		"""
		self.Spline_op = Spline_op
		self.tmp_fine_model = Spline_op.range.clone()
		return

	def setVel(self,vel_in):
		if("Spline_op" in dir(self)):
			self.Spline_op.forward(False,vel_in,self.tmp_fine_model)
			vel = self.tmp_fine_model
		else:
			vel = vel_in
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.setVel(vel)
		return

#################################### Tomo ######################################
def tomoExtOpInitFloat(args,client=None):

	# IO object
	parObject=genericIO.io(params=args)

	# Velocity model
	velFile=parObject.getString("vel", "noVelFile")
	if (velFile == "noVelFile"):
		print("**** ERROR: User did not provide vel file ****\n")
		quit()
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Space axes
	zAxis=velFloat.getHyper().axes[0]
	xAxis=velFloat.getHyper().axes[1]

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Extension axis
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR: User did not provide extension type ****\n")
		quit()

	nExt=parObject.getInt("nExt", -1)
	if (nExt == -1):
		print("**** ERROR: User did not provide size of extension axis ****\n")
		quit()
	if (nExt%2 ==0):
		print("Length of extension axis must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt=parObject.getFloat("dts",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt=parObject.getFloat("dx",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	extAxis=Hypercube.axis(n=nExt,o=oExt,d=dExt) # Create extended axis

	# Read sources signals (we assume one unique point source signature for all shots)
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)

	# Extended reflectivity
	reflectivityFile=parObject.getString("reflectivity","None")
	if (reflectivityFile=="None"):
		reflectivityFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,extAxis]))
		reflectivityFloat.scale(0.0)
	else:
		reflectivityFloat=genericIO.defaultIO.getVector(reflectivityFile,ndims=3)

	# Allocate model
	modelFloat=SepVector.getSepVector(velFloat.getHyper())
	#Local vector copy useful for dask interface
	modelFloatLocal = modelFloat


	if client:

		#Getting number of workers and passing
		nWrks = client.getNworkers()

		#Spreading velocity model to workers
		hyper_vel = velFloat.getHyper()
		velFloatD = pyDaskVector.DaskVector(client,vectors=[velFloat]*nWrks)
		velFloat = velFloatD.vecDask

		modelFloat = pyDaskVector.DaskVector(client,vectors=[modelFloat]*nWrks)

		reflectivityFloat = pyDaskVector.DaskVector(client,vectors=[reflectivityFloat]*nWrks)

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometryDask(parObject,velFloat,hyper_vel,client)
		receiversVector,receiverAxis=buildReceiversGeometryDask(parObject,velFloat,hyper_vel,client)

		#Spreading source wavelet
		sourcesSignalsFloat = pyDaskVector.DaskVector(client,vectors=[sourcesSignalsFloat]*nWrks).vecDask
		sourcesSignalsVector = client.getClient().map((lambda x: [x]),sourcesSignalsFloat,pure=False)
		daskD.wait(sourcesSignalsVector)

		#Allocating data vectors to be spread
		dataFloat = []
		for iwrk in range(nWrks):
			dataHyper=Hypercube.hypercube(axes=[timeAxis,receiverAxis[iwrk],sourceAxis[iwrk]])
			dataFloat.append(SepVector.getSepVector(dataHyper))
		dataFloat = pyDaskVector.DaskVector(client,vectors=dataFloat,copy=False)

		#Spreading/Instantiating the parameter objects
		parObject = spreadParObj(client,args,parObject)

	else:

		# Build sources/receivers geometry
		sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
		receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

		sourcesSignalsVector=[]
		sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

		# Allocate data
		dataFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis]))

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,reflectivityFloat,modelFloatLocal

class tomoExtShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector,reflectivityExt):
		# Domain = Background perturbation
		# Range = Born data
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		if("getCpp" in dir(reflectivityExt)):
			reflectivityExt = reflectivityExt.getCpp()

		self.pyOp = pyAcoustic_iso_float_tomo.tomoExtShotsGpu(velocity,paramP.param,sourceVector,sourcesSignalsVector,receiversVector,reflectivityExt)
		return

	def __str__(self):
		"""Name of the operator"""
		return " TomoOp "

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def setReflectivityExt(self,reflectivityExt):
		#Checking if getCpp is present
		if("getCpp" in dir(reflectivityExt)):
			reflectivityExt = reflectivityExt.getCpp()
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			self.pyOp.setReflectivityExt(reflectivityExt)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_tomo.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

# ################################### Wemva ####################################
def wemvaExtOpInitFloat(args):

	# IO object
	parObject=genericIO.io(params=args)

	# Velocity model
	velFile=parObject.getString("vel", "noVelFile")
	if (velFile == "noVelFile"):
		print("**** ERROR: User did not provide vel file ****\n")
		quit()
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,velFloat)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,velFloat)

	# Space axes
	zAxis=velFloat.getHyper().axes[0]
	xAxis=velFloat.getHyper().axes[1]

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Extension axis
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR: User did not provide extension type ****\n")
		quit()

	nExt=parObject.getInt("nExt", -1)
	if (nExt == -1):
		print("**** ERROR: User did not provide size of extension axis ****\n")
		quit()
	if (nExt%2 ==0):
		print("Length of extension axis must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt=parObject.getFloat("dts",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt=parObject.getFloat("dx",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	extAxis=Hypercube.axis(n=nExt,o=oExt,d=dExt) # Create extended axis

	# Read sources signals (we assume one unique point source signature for all shots)
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

	# Receiver signals (Seismic data or "wemvaData") as a float3DReg
	wemvaDataFile=parObject.getString("seismicData","noWemvaDataFile")
	if (wemvaDataFile == "noWemvaDataFile"):
		print("**** ERROR: User did not provide wemva seismic data file ****\n")
		quit()
	receiversSignalsFloat=genericIO.defaultIO.getVector(wemvaDataFile,ndims=3) 	# Read seismic data as a 3DReg
	receiversSignalsFloatNp=receiversSignalsFloat.getNdArray() # Get the numpy array of the total dataset

	# Initialize receivers signals vector
	receiversSignalsVector=[]

	# Copy wemva data to vector of 2DReg
	for iShot in range(sourceAxis.n):
		receiversSignalsSliceFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis])) # Create a 2DReg data slice
		receiversSignalsSliceFloatNp=receiversSignalsSliceFloat.getNdArray() # Get the numpy array of the slice
		for iReceiver in range(receiverAxis.n):
			for its in range(timeAxis.n):
				receiversSignalsSliceFloatNp[iReceiver][its]=receiversSignalsFloatNp[iShot][iReceiver][its]

		# Push back slice to vector after each shot
		receiversSignalsVector.append(receiversSignalsSliceFloat)

	# Allocate data
	dataFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,extAxis]))

	# Allocate model
	modelFloat=SepVector.getSepVector(velFloat.getHyper())

	return modelFloat,dataFloat,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,receiversSignalsVector

class wemvaExtShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector,receiversSignalsVector):
		#Domain = source wavelet
		#Range = recorded data space
		self.setDomainRange(domain,range)
		#Checking if getCpp is present
		if("getCpp" in dir(velocity)):
			velocity = velocity.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()

		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()
		for idx,receiversSignal in enumerate(receiversSignalsVector):
			if("getCpp" in dir(receiversSignal)):
				receiversSignalsVector[idx] = receiversSignal.getCpp()
		self.pyOp = pyAcoustic_iso_float_wemva.wemvaExtShotsGpu(velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector,receiversSignalsVector)
		return

	def __str__(self):
		"""Name of the operator"""
		return " WemvaOp"

	def forward(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.forward(add,model,data)
		return

	def adjoint(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.adjoint(add,model,data)
		return

	def forwardWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.forwardWavefield(add,model,data)
		return

	def adjointWavefield(self,add,model,data):
		#Checking if getCpp is present
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.adjointWavefield(add,model,data)
		return

	def setVel(self,vel):
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			self.pyOp.setVel(vel)
		return

	def dotTestCpp(self,verb=False,maxError=.00001):
		"""Method to call the Cpp class dot-product test"""
		with pyAcoustic_iso_float_wemva.ostream_redirect():
			result=self.pyOp.dotTest(verb,maxError)
		return result

############################## Wemva nonlinear #################################
def wemvaNonlinearOpInitFloat(args):

	# IO object
	parObject=genericIO.io(params=args)

	# Model (velocity)
	modelFile=parObject.getString("vel", "noVelFile")
	if (modelFile == "noVelFile"):
		print("**** ERROR: User did not provide vel file ****\n")
		quit()
	modelFloat=genericIO.defaultIO.getVector(modelFile)

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,modelFloat)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,modelFloat)

	# Non-extended spatial axes
	zAxis=modelFloat.getHyper().axes[0]
	xAxis=modelFloat.getHyper().axes[1]

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Extension axis
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR: User did not provide extension type ****\n")
		quit()

	nExt=parObject.getInt("nExt", -1)
	if (nExt == -1):
		print("**** ERROR: User did not provide size of extension axis ****\n")
		quit()
	if (nExt%2 ==0):
		print("Length of extension axis must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt=parObject.getFloat("dts",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt=parObject.getFloat("dx",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	extAxis=Hypercube.axis(n=nExt,o=oExt,d=dExt) # Create extended axis

	# Read sources signals (we assume one unique point source signature for all shots)
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometry(parObject,modelFloat)
	receiversVector,receiverAxis=buildReceiversGeometry(parObject,modelFloat)

	# Allocate data (image)
	imageFloat=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,extAxis]))

	# Allocate seismic data
	seismicDataFile=parObject.getString("seismicData", "noSeismicDataFile")
	print("seismicDataFile",seismicDataFile)
	if (seismicDataFile == "noSeismicDataFile"):
		print("**** ERROR: User did not provide seismic data file ****\n")
		quit()
	seismicDataFloat=genericIO.defaultIO.getVector(seismicDataFile)

	return modelFloat,imageFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,seismicDataFloat

class wemvaNonlinearShotsGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Extended Born operator"""

	def __init__(self,domain,range,paramP,sourceVector,sourcesSignalsVector,receiversVector,seismicData):
		# Domain = velocity model
		# Range = Extended image
		self.setDomainRange(domain,range)
		print("seismicData",type(seismicData))
		if("getCpp" in dir(domain)):
			domain = domain.getCpp()
		if("getCpp" in dir(paramP)):
			paramP = paramP.getCpp()
		if("getCpp" in dir(seismicData)):
			seismicData = seismicData.getCpp()
			self.seismicData = seismicData.clone()
		for idx,sourceSignal in enumerate(sourcesSignalsVector):
			if("getCpp" in dir(sourceSignal)):
				sourcesSignalsVector[idx] = sourceSignal.getCpp()

		# Instanciate Born ext adjoint
		self.pyOp = pyAcoustic_iso_float_born_ext.BornExtShotsGpu(domain,paramP,sourceVector,sourcesSignalsVector,receiversVector)
		return

	def __str__(self):
		"""Name of the operator"""
		return " WemvaOp"

	def forward(self,add,model,data):
		# Model = velocity
		# Data = migrated image
		self.setVel(model)
		if("getCpp" in dir(model)):
			model = model.getCpp()
		if("getCpp" in dir(data)):
			data = data.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.adjoint(add,data,self.seismicData)
		return

	def add_spline(self,Spline_op):
		"""
		   Adding spline operator to set background
		"""
		self.Spline_op = Spline_op
		self.tmp_fine_model = Spline_op.range.clone()
		return

	def setVel(self,vel_in):
		if("Spline_op" in dir(self)):
			self.Spline_op.forward(False,vel_in,self.tmp_fine_model)
			vel = self.tmp_fine_model
		else:
			vel = vel_in
		#Checking if getCpp is present
		if("getCpp" in dir(vel)):
			vel = vel.getCpp()
		with pyAcoustic_iso_float_born_ext.ostream_redirect():
			self.pyOp.setVel(vel)
		return


############################## Symes' pseudo-inverse ###########################
def SymesPseudoInvInit(args):

	# IO object
	parObject=genericIO.io(params=args)

	############################## Born extended ###############################
	# Velocity model
	velFile=parObject.getString("vel", "noVelFile")
	if (velFile == "noVelFile"):
		print("**** ERROR: User did not provide vel file ****\n")
		quit()
	velFloat=genericIO.defaultIO.getVector(velFile)

	# Space axes
	zAxis=velFloat.getHyper().axes[0]
	xAxis=velFloat.getHyper().axes[1]
	fat=parObject.getInt("fat")
	taperEndTraceWidth=parObject.getFloat("taperEndTraceWidth",0.5)

	# Time axis
	nts=parObject.getInt("nts",-1)
	ots=parObject.getFloat("ots",0.0)
	dts=parObject.getFloat("dts",-1.0)
	timeAxis=Hypercube.axis(n=nts,o=ots,d=dts)

	# Extension axis
	extension=parObject.getString("extension", "noExtensionType")
	if (extension == "noExtensionType"):
		print("**** ERROR: User did not provide extension type ****\n")
		quit()

	nExt=parObject.getInt("nExt", -1)
	if (nExt == -1):
		print("**** ERROR: User did not provide size of extension axis ****\n")
		quit()
	if (nExt%2 ==0):
		print("Length of extension axis must be an uneven number")
		quit()

	# Time extension
	if (extension == "time"):
		dExt=parObject.getFloat("dts",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	# Horizontal subsurface offset extension
	if (extension == "offset"):
		dExt=parObject.getFloat("dx",-1.0)
		hExt=(nExt-1)/2
		oExt=-dExt*hExt

	extAxis=Hypercube.axis(n=nExt,o=oExt,d=dExt) # Create extended axis

	# Read sources signals (we assume one unique point source signature for all shots)
	sourcesFile=parObject.getString("sources","noSourcesFile")
	if (sourcesFile == "noSourcesFile"):
		print("**** ERROR: User did not provide seismic sources file ****\n")
		quit()
	sourcesSignalsFloat=genericIO.defaultIO.getVector(sourcesFile,ndims=2)
	sourcesSignalsVector=[]
	sourcesSignalsVector.append(sourcesSignalsFloat) # Create a vector of float2DReg slices

	# Build sources/receivers geometry
	sourcesVector,sourceAxis=buildSourceGeometryDipole(parObject,velFloat)
	receiversVector,receiverAxis=buildReceiversGeometryDipole(parObject,velFloat)

	# Allocate data (extended image: output of pseudo inverse operator)
	data=SepVector.getSepVector(Hypercube.hypercube(axes=[zAxis,xAxis,extAxis]))

	# Allocate model (seismic data: input of pseudo inverse operator)
	model=SepVector.getSepVector(Hypercube.hypercube(axes=[timeAxis,receiverAxis,sourceAxis]))

	return model,data,velFloat,parObject,sourcesVector,sourcesSignalsVector,receiversVector,dts,fat,taperEndTraceWidth

class SymesPseudoInvGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Extended Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector,dts,fat,taperEndTraceWidth):
		# Domain = Seismic data
		# Range = Extended image
		self.setDomainRange(domain,range)
		# Instanciate data taper for end of trace
		self.dataTaperOp=dataTaperModule.datTaper(domain,domain,0,0,0,0,0,0,0,0,0,0,domain.getHyper(),0,0,0,0,0,0,0,0,0,taperEndTraceWidth)
		# Instanciate Born extended (with dipole)
		self.BornExtOp=BornExtShotsGpu(range,domain,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector)
		# Instanciate 3rd time integral
		self.timeIntegOp=timeIntegModule.timeInteg(domain,dts)
		# Instanciate Symes z-derivative
		self.SymesZGradOp=spatialDerivModule.SymesZGradPython(range,fat)
		#Allocate temporary vectors
		self.tmp1 = domain.clone() #Output for time integration
		self.tmp2 = domain.clone() #Output for time integration
		self.tmp3 = range.clone()  #Output for Born extended adjoint with dipole
		self.tmp4 = range.clone()  #Output for z derivative

		return

	def forward(self,add,model,data):
		# Apply time integral (x3)
		self.timeIntegOp.forward(False,model,self.tmp1)
		# Apply trace tapering
		self.dataTaperOp.forward(False,self.tmp1,self.tmp2)
		# Apply Born extended with dipole
		self.BornExtOp.adjoint(False,self.tmp3,self.tmp2)
		# Apply z-derivative
		self.SymesZGradOp.forward(False,self.tmp3,self.tmp4)
		# Scale by 8*velocity^4
		vel = self.BornExtOp.getVel()
		velNd = vel.getNdArray()
		vel_tmp = np.expand_dims(velNd,axis=0)
		tmp4Nd = self.tmp4.getNdArray()
		tmp4Nd = tmp4Nd*8.0*vel_tmp*vel_tmp*vel_tmp*vel_tmp
		if(not add):
			dataNd=data.getNdArray()
			dataNd[:]=tmp4Nd[:]
		else:
			data.scaleAdd(self.tmp4)
		return

	def setVel(self,vel):
		self.BornExtOp.setVel(vel)
		return

################################ Symes' Wd + Born-extended #####################
class SymesWdBornExtGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Extended Born operator"""

	def __init__(self,domain,range,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector,dts,fat,taperEndTraceWidth):
		# Domain = Seismic data
		# Range = Extended image
		self.setDomainRange(domain,range)
		# Instanciate Born extended (with dipole)
		self.BornExtOp=BornExtShotsGpu(range,domain,velocity,paramP,sourceVector,sourcesSignalsVector,receiversVector)
		# Instanciate 3rd time integral
		self.timeIntegOp=timeIntegModule.timeInteg(domain,dts)
		#Allocate temporary vectors
		self.tmp1 = domain.clone() # Output for time integration
		self.tmp2 = range.clone() # Output for Born extended
		return

	def forward(self,add,model,data):
		# Apply time integral (x3)
		self.timeIntegOp.forward(False,model,self.tmp1)
		# Apply Born extended with dipole
		self.BornExtOp.adjoint(False,self.tmp2,self.tmp1)
		tmp2Nd = self.tmp2.getNdArray()
		if(not add):
			dataNd=data.getNdArray()
			dataNd[:]=tmp2Nd[:]
		else:
			data.scaleAdd(self.tmp2)
		return

	def setVel(self,vel):
		self.BornExtOp.setVel(vel)
		return

################################ Symes' Wd #####################################
class SymesWdGpu(Op.Operator):
		"""Wrapper encapsulating PYBIND11 module for Wd"""

		def __init__(self,domain,dts):
				# Domain = Seismic data
				# Range = Extended image
				self.setDomainRange(domain,domain)
				# Instanciate 3rd time integral
				self.timeIntegOp=timeIntegModule.timeInteg(domain,dts)
				return

		def forward(self,add,model,data):
				self.checkDomainRange(model,data)
				# Apply time integral (x3)
				self.timeIntegOp.forward(add,model,data)
				return

################################ Symes' Wm #####################################
class SymesWmGpu(Op.Operator):
	"""Wrapper encapsulating PYBIND11 module for Extended Born operator"""

	def __init__(self,domain,range,velocity,fat):

		# Domain = Seismic data
		# Range = Extended image
		self.setDomainRange(domain,range)

		# Instanciate Symes z-derivative
		self.SymesZGradOp=spatialDerivModule.SymesZGradPython(range,fat)

		# Set velocity value
		self.vel=velocity

		#Allocate temporary vectors
		self.tmp1 = domain.clone() #Output for time integration

		return

	def forward(self,add,model,data):

		self.SymesZGradOp.forward(False,model,self.tmp1)
		# Scale by 8*velocity^4
		velNd = self.vel.getNdArray()
		vel_tmp = np.expand_dims(velNd,axis=0)
		tmp1Nd = self.tmp1.getNdArray()
		tmp1Nd = tmp1Nd*8.0*vel_tmp*vel_tmp*vel_tmp*vel_tmp
		if(not add):
			# dataNd=data.getNdArray()
			# dataNd[:]=tmp4Nd[:]
			data.copy(self.tmp1)
		else:
			data.scaleAdd(self.tmp1)
		return

	def setVel(self,vel):
		self.vel=vel
		return

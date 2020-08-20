#Python module encapsulating PYBIND11 module
import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np
from numpy import linalg as LA

################################################################################
############################### Nonlinear forward ##############################
################################################################################
# Nonlinear operator
class phaseOnlyXk(Op.Operator):

	def __init__(self,domain,range):

		self.setDomainRange(domain,range)
		return

	def forward(self,add,model,data):
		# Check domain/range
		self.checkDomainRange(model,data)
		modelNp=model.getNdArray()
		dataNp=data.getNdArray()
		if (not add):
			data.zero()
		# Loop over shots and receivers
		nShot = model.getHyper().getAxis(3).n
		nReceiver = model.getHyper().getAxis(2).n
		for iShot in range(nShot):
			for iReceiver in range(nReceiver):
				# Compute model norm for this trace
				modelNormInv=LA.norm(modelNp[iShot][iReceiver][:])
				modelNormInv=1/modelNormInv
				# Compute normalized data (trace)
				dataNp[iShot][iReceiver][:]+=modelNp[iShot][iReceiver][:]*modelNormInv

		return

################################################################################
############################### Jacobian operator ##############################
################################################################################
def phaseOnlyXkJacInit(args):

	# IO object
	parObject=genericIO.io(params=args)

	# Allocate and read predicted data f(m) (i.e, the "background" data)
	predDataFile=parObject.getString("predData")
	predDat=genericIO.defaultIO.getVector(predDataFile,ndims=3)
	predDatNp=predDat.getNdArray()

	return predDat

class phaseOnlyXkJac(Op.Operator):

	def __init__(self,predDat):

		# Set domain/range (same size as observed/predicted data)
		self.setDomainRange(predDat,predDat)
		self.predDat=predDat
		return

	def forward(self,add,model,data):
		# Check domain/range
		self.checkDomainRange(model,data)
		modelNp=model.getNdArray()
		dataNp=data.getNdArray()
		predDatNp=self.predDat.getNdArray()
		if (not add):
			data.zero()

		# Loop over shots and receivers
		nShot = model.getHyper().getAxis(3).n
		nReceiver = model.getHyper().getAxis(2).n
		for iShot in range(nShot):
			for iReceiver in range(nReceiver):

				# Compute inverse of predicted trace norm
				predDatNormInv=LA.norm(predDatNp[iShot][iReceiver][:])
				predDatNormInv=1/predDatNormInv
				# Compute cube of inverse of predicted trace norm
				predDatNormCubeInv=predDatNormInv*predDatNormInv*predDatNormInv
				# Compute dot product between model and predicted trace
				dotProdDatMod=np.dot(predDatNp[iShot][iReceiver][:],modelNp[iShot][iReceiver][:])
				# Apply forward
				dataNp[iShot][iReceiver][:]+=modelNp[iShot][iReceiver][:]*predDatNormInv-dotProdDatMod*predDatNormCubeInv*predDatNp[iShot][iReceiver][:]

		return

	def adjoint(self,add,model,data):
		# Check domain/range
		self.checkDomainRange(model,data)
		modelNp=model.getNdArray()
		dataNp=data.getNdArray()
		predDatNp=self.predDat.getNdArray()
		if (not add):
			model.zero()

		# Loop over shots and receivers
		nShot = model.getHyper().getAxis(3).n
		nReceiver = model.getHyper().getAxis(2).n
		for iShot in range(nShot):
			for iReceiver in range(nReceiver):

				# Compute inverse of predicted trace norm
				predDatNormInv=LA.norm(predDatNp[iShot][iReceiver][:])
				predDatNormInv=1/predDatNormInv
				# Compute cube of inverse of predicted trace norm
				predDatNormCubeInv=predDatNormInv*predDatNormInv*predDatNormInv
				# Compute dot product between model and predicted trace
				dotProdDatMod=np.dot(predDatNp[iShot][iReceiver][:],dataNp[iShot][iReceiver][:])
				# Apply forward
				modelNp[iShot][iReceiver][:]+=dataNp[iShot][iReceiver][:]*predDatNormInv-dotProdDatMod*predDatNormCubeInv*predDatNp[iShot][iReceiver][:]

		return

	def setData(self,data):
		# Pointer assignement
		self.predDat=data

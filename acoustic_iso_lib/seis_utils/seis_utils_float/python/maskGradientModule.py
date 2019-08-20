#Python module encapsulating PYBIND11 module
import pyOperator as Op
import genericIO
import SepVector
import Hypercube
import numpy as np

def maskGradientInit(args):

	io=genericIO.pyGenericIO.ioModes(args)
	ioDef=io.getDefaultIO()
	parObject=ioDef.getParamObj()

	# Check if user directly provides the mask for the gradient
	gradientMaskFile=parObject.getString("gradientMaskFile","noGradientMaskFile")

	# Case where the user wants to apply a mask but does not provide the file
	# The gradient is computed automatically by providing the following parameters
	if (gradientMaskFile=="noGradientMaskFile"):
		print("--- User has not provided a gradient mask file ---")
		print("--- Automatically generating the mask from the provided parameters ---")
		# Read parameters
		bufferUp=parObject.getFloat("bufferUp") # Taper width above water bottom [km]
		bufferDown=parObject.getFloat("bufferDown") # Taper width below water bottom [km]
		taperExp=parObject.getFloat("taperExp") # Taper exponent
		fat=parObject.getInt("fat",5)
		velFile=parObject.getString("vel","noVelFile")
		vel=genericIO.defaultIO.getVector(velFile)
		wbShift=parObject.getFloat("wbShift") # Shift water bottom velocity [km] to start tapering at a different depth

	# The user provides the gradient mask file
	# If you provide both (parameters and mask file, the mask file has priority)
	else:
		print("--- User has provided a gradient mask file ---")
		bufferUp=0
		bufferDown=0
		taperExp=0
		fat=0
		velFile=parObject.getString("vel","noVelFile")
		vel=genericIO.defaultIO.getVector(velFile)
		wbShift=0

	return vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile

class maskGradient(Op.Operator):

	def __init__(self,domain,Range,vel,bufferUp,bufferDown,taperExp,fat,wbShift,gradientMaskFile):

		# Set domain/range
		self.setDomainRange(domain,Range)

		# Case where the user wants to apply a mask but does not provide the file
		# The gradient is computed automatically
		if (gradientMaskFile=="noGradientMaskFile"):
			# Create mask and set it to zero
			self.mask=vel.clone()
			self.mask.set(0.0)
			maskNp=self.mask.getNdArray()

			# Compute water velocity (asssumed to be constant)
			velNp=vel.getNdArray()
			waterVel=velNp[fat][fat]

			# Substract water velocity from velocity model
			velNp=velNp-waterVel

			# Compute index of water bottom
			nxVel=vel.getHyper().axes[1].n
			ozVel=vel.getHyper().axes[0].o
			dzVel=vel.getHyper().axes[0].d
			nzVel=vel.getHyper().axes[0].n
			indexWaterBottom=np.zeros((nxVel)) # Water bottom index
			depthWaterBottom=np.zeros((nxVel)) # Water bottom depth [km]
			depthUpper=np.zeros((nxVel)) # Upper bound depth [km]
			depthLower=np.zeros((nxVel)) # Lower bound depth [km]
			indexUpper=np.zeros((nxVel)) # Upper bound index
			indexLower=np.zeros((nxVel)) # Lower bound index

			# Convert water bottom hift from km->samples
			iWbShift=int(wbShift/dzVel)

			for ix in range(nxVel-2*fat):

				# Compute water bottom index
				indexWaterBottom[ix+fat]=np.argwhere(velNp[ix+fat][:]>0)[0][0]
				indexWaterBottom[ix+fat]=indexWaterBottom[ix+fat]-1+iWbShift

				# Compute water bottom depth [km]
				depthWaterBottom[ix+fat]=ozVel+indexWaterBottom[ix+fat]*dzVel

				# Compute water bottom upper bound
				depthUpper[ix+fat]=depthWaterBottom[ix+fat]-bufferUp # Upper bound [km]
				indexUpper[ix+fat]=(depthUpper[ix+fat]-ozVel)/dzVel # Upper bound [sample]

				# Compute water bottom upper bound [km]
				depthLower[ix+fat]=depthWaterBottom[ix+fat]+bufferDown # Lower bound [km]
				indexLower[ix+fat]=(depthLower[ix+fat]-ozVel)/dzVel # Lower bound [sample]

				iz1=int(indexUpper[ix+fat])
				iz2=int(indexLower[ix+fat])

				# Compute weight
				for iz in range(iz1,iz2):
					weight=(iz-iz1)/(iz2-iz1)
					weight=np.sin(np.pi*0.5*weight)
					maskNp[ix+fat][iz]=np.power(weight,taperExp)

				maskNp[ix+fat][iz2:]=1.0

		# Case where the user wants to apply a mask and provides the file for the mask
		else:
			self.mask=genericIO.defaultIO.getVector(gradientMaskFile,ndims=2)

		return

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		modelNp=model.getNdArray()
		maskNp=self.mask.getNdArray()
		if (not add):
			data.zero()
		dataNp=data.getNdArray()
		# dataNp+=modelNp*maskNp
		dataNp+=modelNp

		return

	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		dataNp=data.getNdArray()
		maskNp=self.mask.getNdArray()
		if (not add):
			model.zero()
		modelNp=model.getNdArray()
		modelNp+=dataNp*maskNp

		return

	def getMask(self):
		return self.mask

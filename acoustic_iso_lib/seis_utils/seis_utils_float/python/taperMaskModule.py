#Python module encapsulating PYBIND11 module
import pyOperator as Op
import SepVector
import numpy as np

class taperMask(Op.Operator):
	"""Operator class to apply a mask to an array"""

	def __init__(self,domain,mask):
		self.setDomainRange(domain,domain)
		self.mask.checkSame(domain)
        self.mask_array = mask.getNdArray()
		return

	def forward(self,add,model,data):
		"""Method to compute d = Mask m"""
		self.checkDomainRange(model,data)
		if(not add): data.zero()
		data.getNdArray()[:]+= model.getNdArray()[:]*self.mask_array[:]
		return

    def adjoint(self,add,model,data):
        """Method to compute m = Mask d"""
		self.checkDomainRange(model,data)
		if(not add): data.zero()
		model.getNdArray()[:]+= data.getNdArray()[:]*self.mask_array[:]
		return

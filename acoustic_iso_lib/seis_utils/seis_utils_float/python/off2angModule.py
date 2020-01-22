#Python module for subsurface offsets to angle conversion
import pyOperator as Op
import SepVector
import numpy as np


class off2ang2D(Op.Operator):
	"""Operator class to apply a mask to an array"""


	def __init__(self,domain,range,oz,dz,oh,dh,og,dg):
		"""
		   Operator to convert imagese from/to subsurface offsets to/from angles
		   :param domain: vector class, Vector defining the size of the subsurface-offset-domain image (z,x,h)
		   :param oz: int, Origin of the z axis
		   :param dz: int, Sampling of the z axis
		   :param oh: int, Origin of the subsurface-offset axis
		   :param dh: int, Sampling of the subsurface-offset axis
		   :param og: int, Origin of the angle axis in degree
		   :param dg: int, Sampling of the angle axis in degree
		"""
		self.setDomainRange(domain,range)
		# Number of elements
		self.nz = domain.getNdArray().shape[-1]
		if self.nz != range.getNdArray().shape[-1]:
			raise ValueError("Number of element of the image spaces must be the same (z samples: offset %s, angle %s )"%(self.nz, range.getNdArray().shape[-1]))
		self.nh = domain.getNdArray().shape[0]
		self.ng = range.getNdArray().shape[0]
		# Origins
		self.oz = oz
		self.oh = oh
		self.og = og*np.pi/180.0
		# Sampling
		self.dz = dz
		self.dh = dh
		self.dg = dg*np.pi/180.0

	def forward(self,add,model,data):
		"""Method to convert extended image from h to angles"""
		self.checkDomainRange(model,data)
		if not add:
			data.zero()

		# Getting Nd arrays
		m_arr = model.getNdArray()
		d_arr = data.getNdArray()
		d_tmp = np.zeros(d_arr.shape, dtype=complex)

		# kz sampling information
		dkz = 1.0/((self.nz)*self.dz)
		kz_axis = np.linspace(0.0,(self.nz-1)*dkz,self.nz)

		m_kz = np.fft.fft(m_arr)
		# Precomputing scaling factor for transformation
		h_axis = np.linspace(self.oh,self.oh+(self.nh-1)*self.dh,self.nh)
		exp_arg = np.expand_dims(1.0j*np.outer(h_axis,kz_axis),axis=1)
		tan_vals = np.tan(np.linspace(self.og,self.og+(self.ng-1)*self.dg,self.ng))
		for g_idx,tg_v in enumerate(tan_vals):
			d_tmp[g_idx,:,:] = np.sum(m_kz[:,:,:]*np.exp(exp_arg*tg_v),axis=0)*self.dh
		d_arr += np.real(np.fft.ifft(d_tmp))
		return

	def adjoint(self,add,model,data):
		"""Method to convert extended image from angles to h"""
		self.checkDomainRange(model,data)
		if not add:
			model.zero()
		return

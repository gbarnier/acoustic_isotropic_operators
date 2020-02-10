#Python module for subsurface offsets to angle conversion
import pyOperator as Op
import SepVector
import numpy as np


class off2ang2D(Op.Operator):
	"""Operator class to transform ADCIGs to ODCIGs and vice versa"""


	def __init__(self,domain,range,oz,dz,oh,dh,og,dg,p_inv=True):
		"""
		   Operator to convert imagese from/to angles to/from subsurface offsets
		   :param domain: vector class, Vector defining the size of the angle-domain image (z,x,gamma)
		   :param range: vector class, Vector defining the size of the subsurface-offset-domain image (z,x,h)
		   :param oz: int, Origin of the z axis
		   :param dz: int, Sampling of the z axis
		   :param oh: int, Origin of the subsurface-offset axis
		   :param dh: int, Sampling of the subsurface-offset axis
		   :param og: int, Origin of the angle axis in degree
		   :param dg: int, Sampling of the angle axis in degree
		   :param p_inv: boolean, whether to apply pseudo-inverse or simple adjoint operator [True]
		"""
		self.setDomainRange(domain,range)
		# Number of elements
		self.nz = domain.getNdArray().shape[-1]
		if self.nz != range.getNdArray().shape[-1]:
			raise ValueError("Number of element of the image spaces must be the same (z samples: offset %s, angle %s )"%(self.nz, range.getNdArray().shape[-1]))
		self.ng = domain.getNdArray().shape[0]
		self.nh = range.getNdArray().shape[0]
		# Origins
		self.oz = oz
		self.oh = oh
		self.og = og*np.pi/180.0
		# Sampling
		self.dz = dz
		self.dh = dh
		self.dg = dg*np.pi/180.0
		self.p_inv = p_inv

	def forward(self,add,model,data):
		"""Method to convert extended image from angles to h"""
		self.checkDomainRange(model,data)
		if not add:
			data.zero()

		# Getting Nd arrays
		m_arr = model.getNdArray()
		d_arr = data.getNdArray()
		d_tmp = np.zeros(d_arr.shape, dtype=complex)

		# kz sampling information
		kz_axis = 2.0*np.pi*np.fft.fftfreq(self.nz,self.dz)

		# Fourier transform of input ADCIGs
		m_kz = np.fft.fft(m_arr)

		# Precomputing scaling factor for transformation
		h_axis = np.linspace(self.oh,self.oh+(self.nh-1)*self.dh,self.nh)
		g_vals = np.linspace(self.og,self.og+(self.ng-1)*self.dg,self.ng)
		exp_arg = np.expand_dims(-1.0j*np.outer(np.tan(g_vals),kz_axis),axis=1)
		for h_idx,off_val in enumerate(h_axis):
			d_tmp[h_idx,:,:] = np.sum(m_kz[:,:,:]*np.exp(exp_arg*off_val),axis=0)
		d_arr += np.real(np.fft.ifft(d_tmp))
		return

	def adjoint(self,add,model,data):
		"""Method to convert extended image from h to angles"""
		self.checkDomainRange(model,data)
		if not add:
			model.zero()

		# Getting Nd arrays
		m_arr = model.getNdArray() # ADCIGs
		d_arr = data.getNdArray()  # ODCIGs
		m_tmp = np.zeros(m_arr.shape, dtype=complex)

		# kz sampling information
		kz_axis = 2.0*np.pi*np.fft.fftfreq(self.nz,self.dz)
		dkz = kz_axis[1]-kz_axis[0]

		# Fourier transform of input ODCIGs
		d_kz = np.fft.fft(d_arr)

		# Precomputing scaling factor for transformation
		h_axis = np.linspace(self.oh,self.oh+(self.nh-1)*self.dh,self.nh)
		g_vals = np.linspace(self.og,self.og+(self.ng-1)*self.dg,self.ng)
		exp_arg = np.expand_dims(1.0j*np.outer(h_axis,kz_axis),axis=1)
		for g_idx,g_val in enumerate(g_vals):
			# scale = 2.0*np.pi*kz_axis/np.cos(g_val) if self.p_inv else 1.0
			scale = 2.0*np.pi/np.cos(g_val) if self.p_inv else 1.0
			# print("angle=%s cos=%s"%(g_val,np.cos(g_val)))
			m_tmp[g_idx,:,:] = scale*np.sum(d_kz[:,:,:]*np.exp(exp_arg*np.tan(g_val)),axis=0)
		m_arr += np.real(np.fft.ifft(m_tmp))
		return

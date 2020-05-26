import SepVector
import pyOperator as Op
import numpy as np

class lapla2D(Op.Operator):
	def __init__(self,model,ds=1.0):
		self.setDomainRange(model,model)
		self.ds = ds
	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if not add:
			data.zero()
		m_arr = model.getNdArray()
		d_arr = data.getNdArray()
		m_arr[0,:]=0.0
		m_arr[-1,:]=0.0
		m_arr[:,0]=0.0
		m_arr[:,-1]=0.0
		for idx in range(1,m_arr.shape[0]-1):
			for idz in range(1,m_arr.shape[1]-1):
				d_arr[idx,idz] += (m_arr[idx,idz-1]+m_arr[idx,idz+1]-4.0*m_arr[idx,idz]+m_arr[idx-1,idz]+m_arr[idx+1,idz])/(self.ds*self.ds)
		return
	def adjoint(self,add,model,data):
		self.forward(add,data,model)
		return

class scaleVel(Op.Operator):
	def __init__(self,wavefield,vel):
		self.setDomainRange(wavefield,wavefield)
		self.vel = vel.clone()
		self.velScale = self.vel.getNdArray()*self.vel.getNdArray()

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if not add:
			data.zero()
		m_arr = model.getNdArray()
		d_arr = data.getNdArray()
		d_arr[:,:,:] += m_arr[:,:,:]*self.velScale

	def adjoint(self,add,model,data):
		self.forward(add,data,model)
		return

class waveEquation1(Op.Operator):
	def __init__(self,wavefield,vel,dt,ds):
		self.setDomainRange(wavefield,wavefield)
		self.vel = vel
		self.dt = dt
		self.dt2 = dt*dt
		self.ds2_inv = 1.0/(ds*ds)
		self.vel2dt = self.vel.getNdArray()*self.vel.getNdArray()*self.dt2

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if not add:
			data.zero()
		m_arr = model.clone().getNdArray()
		# Scaling input wavefield
		m_arr *= self.vel2dt
		data_tmp = data.clone().zero()
		d_arr = data_tmp.getNdArray()
		nt = m_arr.shape[0]
		nx = m_arr.shape[1]
		nz = m_arr.shape[2]
		m_arr[:,0,:]=0.0
		m_arr[:,-1,:]=0.0
		m_arr[:,:,0]=0.0
		m_arr[:,:,-1]=0.0
		for it in range(nt):
			if it > 1:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						d_arr[it,idx,idz] += (d_arr[it-1,idx,idz-1]+d_arr[it-1,idx,idz+1]-4.0*d_arr[it-1,idx,idz]+d_arr[it-1,idx-1,idz]+d_arr[it-1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*d_arr[it-1,idx,idz] - d_arr[it-2,idx,idz] + m_arr[it-1,idx,idz]
			elif it == 1:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						d_arr[it,idx,idz] += (d_arr[it-1,idx,idz-1]+d_arr[it-1,idx,idz+1]-4.0*d_arr[it-1,idx,idz]+d_arr[it-1,idx-1,idz]+d_arr[it-1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*d_arr[it-1,idx,idz] + m_arr[it-1,idx,idz]
		data.scaleAdd(data_tmp)


	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if not add:
			model.zero()
		d_arr = data.clone().getNdArray()
		model_tmp = model.clone().zero()
		m_arr = model_tmp.getNdArray()
		nt = m_arr.shape[0]
		nx = m_arr.shape[1]
		nz = m_arr.shape[2]
		d_arr[:,0,:]=0.0
		d_arr[:,-1,:]=0.0
		d_arr[:,:,0]=0.0
		d_arr[:,:,-1]=0.0
		for it in range(nt,-1,-1):
			if it < nt-2:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						m_arr[it,idx,idz] += (m_arr[it+1,idx,idz-1]*self.vel2dt[idx,idz+1]+m_arr[it+1,idx,idz+1]*self.vel2dt[idx,idz+1]-4.0*m_arr[it+1,idx,idz]*self.vel2dt[idx,idz]+m_arr[it+1,idx-1,idz]*self.vel2dt[idx-1,idz]+m_arr[it+1,idx+1,idz]*self.vel2dt[idx+1,idz])*self.ds2_inv + 2.0*m_arr[it+1,idx,idz] - m_arr[it+2,idx,idz] + d_arr[it+1,idx,idz]
			elif it == nt-2:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						m_arr[it,idx,idz] += (m_arr[it+1,idx,idz-1]*self.vel2dt[idx,idz-1]+m_arr[it+1,idx,idz+1]*self.vel2dt[idx,idz+1]-4.0*m_arr[it+1,idx,idz]*self.vel2dt[idx,idz]+m_arr[it+1,idx-1,idz]*self.vel2dt[idx-1,idz]+m_arr[it+1,idx+1,idz]*self.vel2dt[idx+1,idz])*self.ds2_inv + 2.0*m_arr[it+1,idx,idz] + d_arr[it+1,idx,idz]
		# Scaling the output
		m_arr *= self.vel2dt
		model.scaleAdd(model_tmp)

class waveEquation2(Op.Operator):
	def __init__(self,wavefield,vel,dt,ds):
		self.setDomainRange(wavefield,wavefield)
		self.vel = vel
		self.dt = dt
		self.dt2 = dt*dt
		self.ds2_inv = 1.0/(ds*ds)
		self.vel2dt = self.vel.getNdArray()*self.vel.getNdArray()*self.dt2

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if not add:
			data.zero()
		m_arr = model.clone().getNdArray()
		# Scaling input wavefield
		m_arr *= self.vel2dt
		data_tmp = data.clone().zero()
		d_arr = data_tmp.getNdArray()
		nt = m_arr.shape[0]
		nx = m_arr.shape[1]
		nz = m_arr.shape[2]
		m_arr[:,0,:]=0.0
		m_arr[:,-1,:]=0.0
		m_arr[:,:,0]=0.0
		m_arr[:,:,-1]=0.0
		for it in range(nt):
			if it > 1:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						d_arr[it,idx,idz] += (d_arr[it-1,idx,idz-1]+d_arr[it-1,idx,idz+1]-4.0*d_arr[it-1,idx,idz]+d_arr[it-1,idx-1,idz]+d_arr[it-1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*d_arr[it-1,idx,idz] - d_arr[it-2,idx,idz] + m_arr[it-1,idx,idz]
			elif it == 1:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						d_arr[it,idx,idz] += (d_arr[it-1,idx,idz-1]+d_arr[it-1,idx,idz+1]-4.0*d_arr[it-1,idx,idz]+d_arr[it-1,idx-1,idz]+d_arr[it-1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*d_arr[it-1,idx,idz] + m_arr[it-1,idx,idz]
		data.scaleAdd(data_tmp)


	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if not add:
			model.zero()
		d_arr = data.clone().getNdArray()
		model_tmp = model.clone().zero()
		m_arr = model_tmp.getNdArray()
		d_arr *= self.vel2dt
		nt = m_arr.shape[0]
		nx = m_arr.shape[1]
		nz = m_arr.shape[2]
		d_arr[:,0,:]=0.0
		d_arr[:,-1,:]=0.0
		d_arr[:,:,0]=0.0
		d_arr[:,:,-1]=0.0
		for it in range(nt,-1,-1):
			if it < nt-2:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						m_arr[it,idx,idz] += (m_arr[it+1,idx,idz-1]+m_arr[it+1,idx,idz+1]-4.0*m_arr[it+1,idx,idz]+m_arr[it+1,idx-1,idz]+m_arr[it+1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*m_arr[it+1,idx,idz] - m_arr[it+2,idx,idz] + d_arr[it+1,idx,idz]
			elif it == nt-2:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						m_arr[it,idx,idz] += (m_arr[it+1,idx,idz-1]+m_arr[it+1,idx,idz+1]-4.0*m_arr[it+1,idx,idz]+m_arr[it+1,idx-1,idz]+m_arr[it+1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*m_arr[it+1,idx,idz] + d_arr[it+1,idx,idz]
		model.scaleAdd(model_tmp)

class waveEquation3(Op.Operator):
	def __init__(self,wavelet,data,vel,dt,ds,s_pos,r_pos):
		self.setDomainRange(wavelet,data)
		self.vel = vel
		self.dt = dt
		self.dt2 = dt*dt
		self.ds2_inv = 1.0/(ds*ds)
		self.s_pos = s_pos
		self.r_pos = r_pos
		self.vel2dt = self.vel.getNdArray()*self.vel.getNdArray()*self.dt2

	def forward(self,add,model,data):
		self.checkDomainRange(model,data)
		if not add:
			data.zero()
		nt = model.getNdArray().shape[0]
		nx = self.vel.getNdArray().shape[0]
		nz = self.vel.getNdArray().shape[1]
		m_arr = np.zeros((nt,nx,nz))
		for id_pos in self.s_pos:
			m_arr[:,id_pos[0],id_pos[1]] = model.getNdArray()
		# Scaling input wavefield
		m_arr *= self.vel2dt
		d_arr = np.zeros((nt,nx,nz))
		m_arr[:,0,:]=0.0
		m_arr[:,-1,:]=0.0
		m_arr[:,:,0]=0.0
		m_arr[:,:,-1]=0.0
		for it in range(nt):
			if it > 1:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						d_arr[it,idx,idz] += (d_arr[it-1,idx,idz-1]+d_arr[it-1,idx,idz+1]-4.0*d_arr[it-1,idx,idz]+d_arr[it-1,idx-1,idz]+d_arr[it-1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*d_arr[it-1,idx,idz] - d_arr[it-2,idx,idz] + m_arr[it-1,idx,idz]
			elif it == 1:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						d_arr[it,idx,idz] += (d_arr[it-1,idx,idz-1]+d_arr[it-1,idx,idz+1]-4.0*d_arr[it-1,idx,idz]+d_arr[it-1,idx-1,idz]+d_arr[it-1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*d_arr[it-1,idx,idz] + m_arr[it-1,idx,idz]
		for rec,id_pos in enumerate(self.r_pos):
			data.getNdArray()[rec,:] += d_arr[:,id_pos[0],id_pos[1]]


	def adjoint(self,add,model,data):
		self.checkDomainRange(model,data)
		if not add:
			model.zero()
		nt = model.getNdArray().shape[0]
		nx = self.vel.getNdArray().shape[0]
		nz = self.vel.getNdArray().shape[1]
		m_arr = np.zeros((nt,nx,nz))
		d_arr = np.zeros((nt,nx,nz))
		for rec,id_pos in enumerate(self.r_pos):
			d_arr[:,id_pos[0],id_pos[1]] = data.getNdArray()[rec,:]
		# Scaling input wavefield
		d_arr *= self.vel2dt
		d_arr[:,0,:]=0.0
		d_arr[:,-1,:]=0.0
		d_arr[:,:,0]=0.0
		d_arr[:,:,-1]=0.0
		for it in range(nt,-1,-1):
			if it < nt-2:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						m_arr[it,idx,idz] += (m_arr[it+1,idx,idz-1]+m_arr[it+1,idx,idz+1]-4.0*m_arr[it+1,idx,idz]+m_arr[it+1,idx-1,idz]+m_arr[it+1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*m_arr[it+1,idx,idz] - m_arr[it+2,idx,idz] + d_arr[it+1,idx,idz]
			elif it == nt-2:
				for idx in range(1,m_arr.shape[0]-1):
					for idz in range(1,m_arr.shape[1]-1):
						m_arr[it,idx,idz] += (m_arr[it+1,idx,idz-1]+m_arr[it+1,idx,idz+1]-4.0*m_arr[it+1,idx,idz]+m_arr[it+1,idx-1,idz]+m_arr[it+1,idx+1,idz])*self.ds2_inv*self.vel2dt[idx,idz] + 2.0*m_arr[it+1,idx,idz] + d_arr[it+1,idx,idz]
		for id_pos in self.s_pos:
			model.getNdArray()[:] += m_arr[:,id_pos[0],id_pos[1]]



if __name__ == '__main__':
	# mod = SepVector.getSepVector(ns=[5,5],storage="dataDouble")
	# op = lapla2D(mod)
	# op.dotTest(True)
	nx=40
	nz=50
	nt=20
	wav = SepVector.getSepVector(ns=[nz,nx,nt],storage="dataDouble")
	vel = SepVector.getSepVector(ns=[nz,nx],storage="dataDouble")
	vel.set(2.0)
	vel.getNdArray()[:,20:] = 2.5
	vel.getNdArray()[:,:] += np.random.rand(nx,nz)*0.1
	ds = 0.02
	dt = 0.001
	# print("Testing wave-equation operator 1")
	# prop1 = waveEquation1(wav,vel,dt,ds)
	# prop1.dotTest(True)
	# print("Testing wave-equation operator 2")
	# prop2 = waveEquation2(wav,vel,dt,ds)
	# prop2.dotTest(True)
	# prop2.dotTest(True)
	# prop2.dotTest(True)
	# prop2.dotTest(True)

	wavelet = SepVector.getSepVector(ns=[nt],storage="dataDouble")
	data = SepVector.getSepVector(ns=[nt,3],storage="dataDouble")
	sou_pos = [[5,5]]
	rec_pos = [[5,5],[5,6],[6,6]]
	print("Testing wave-equation operator 3")
	prop3 = waveEquation3(wavelet,data,vel,dt,ds,sou_pos,rec_pos)
	prop3.dotTest(True)
	prop3.dotTest(True)
	prop3.dotTest(True)
	prop3.dotTest(True)

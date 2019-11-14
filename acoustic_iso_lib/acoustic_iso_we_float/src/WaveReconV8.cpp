#include <WaveReconV8.h>

WaveReconV8::WaveReconV8(const std::shared_ptr<SEP::float3DReg>model,
                         const std::shared_ptr<SEP::float3DReg>data,
                         const std::shared_ptr<SEP::float2DReg>slsqModel,
                         float        U_0,
                         float 				alpha,
			                   int					spongeWidth) {
std::cerr << "VERSION 8" << std::endl;
  // ensure model and data dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  // ensure velModel matches fmatial dimensions of model and data
  assert(model->getHyper()->getAxis(1).n ==
         slsqModel->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n ==
         slsqModel->getHyper()->getAxis(2).n);

  _spongeWidth=spongeWidth;

  n1 =model->getHyper()->getAxis(1).n; //z
   n2 =model->getHyper()->getAxis(2).n; //x
   n3 =model->getHyper()->getAxis(3).n; //t

  // set domain and range
  setDomainRange(model, data);

  // set up  lapl operator
  // get fmatial sampling
  _db = model->getHyper()->getAxis(1).d;
  _da = model->getHyper()->getAxis(2).d;
  _dt = model->getHyper()->getAxis(3).d;

  // calculate lapl coefficients
  C0z = -2.927222222 / (_db * _db);
  C1z =  1.666666667 / (_db * _db);
  C2z = -0.238095238 / (_db * _db);
  C3z =  0.039682539 / (_db * _db);
  C4z = -0.004960317 / (_db * _db);
  C5z =  0.000317460 / (_db * _db);

  C0x = -2.927222222 / (_da * _da);
  C1x =  1.666666667 / (_da * _da);
  C2x = -0.238095238 / (_da * _da);
  C3x =  0.039682539 / (_da * _da);
  C4x = -0.004960317 / (_da * _da);
  C5x =  0.000317460 / (_da * _da);

  C0t_10 = -2.927222222 / (_dt * _dt);
  C1t_10 =  1.666666667 / (_dt * _dt);
  C2t_10 = -0.23809524 / (_dt * _dt);
  C3t_10 =  0.03968254 / (_dt * _dt);
  C4t_10 = -0.00496032 / (_dt * _dt);
  C5t_10 =  0.00031746 / (_dt * _dt);

  C0t_8 = -2.84722222222 / (_dt * _dt);
  C1t_8 =  1.600000000 / (_dt * _dt);
  C2t_8 = -0.200000000/ (_dt * _dt);
  C3t_8 =  0.02539682539 / (_dt * _dt);
  C4t_8 = -0.00178571428 / (_dt * _dt);

  C0t_6 = -2.72222222222 / (_dt * _dt);
  C1t_6 =  1.500000000 / (_dt * _dt);
  C2t_6 = -0.150000000/ (_dt * _dt);
  C3t_6 =  0.01111111111 / (_dt * _dt);

  C0t_4 = -2.50000000000 / (_dt * _dt);
  C1t_4 =  1.33333333333 / (_dt * _dt);
  C2t_4 = -0.08333333333 / (_dt * _dt);

  C0t_2 = -2.00000000000 / (_dt * _dt);
  C1t_2 =  1.000000000 / (_dt * _dt);

  setDomainRange(model, data);

    _fatMask.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
                                      data->getHyper()->getAxis(2).n));
    _fatMask->set(0);
    int nz=data->getHyper()->getAxis(1).n;
    int nx=data->getHyper()->getAxis(2).n;
    int nt=data->getHyper()->getAxis(3).n;
    float alphaCos= 0.85;
    for (int ix = FAT; ix < nx-FAT; ix++){
      for (int iz = FAT; iz < nz-FAT; iz++) {
	(*_fatMask->_mat)[ix][iz] = 1;
      }
    }

    //calculate gamma
    _gamma.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
                                      data->getHyper()->getAxis(2).n));
    _gammaSq.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
                                      data->getHyper()->getAxis(2).n));
    _gamma->set(0.0);
    _gammaSq->set(0.0);
    _U_0 = U_0;
    _alpha = alpha;
    //float _U_0 = 0.001;
    //float _alpha = 0.13;
    for (int ix = FAT; ix < nx-FAT; ix++){
      for (int iz = FAT; iz < nz-FAT; iz++) {
	int distToEdge = std::min(std::min(ix-FAT,iz-FAT),std::min(nz-FAT-iz-1,nx-FAT-ix-1));
	if(distToEdge < _spongeWidth){
          float gamma = _U_0/(std::cosh(_alpha*distToEdge)*std::cosh(_alpha*distToEdge));
	  (*_gamma->_mat)[ix][iz] = 2*gamma/_dt;
	  (*_gammaSq->_mat)[ix][iz] = gamma*gamma;
	}
      }
    }
  // set slowness
  _slsq=slsqModel;
}

void WaveReconV8::set_slsq(std::shared_ptr<SEP::float2DReg>slsq){
	_slsq=slsq;
}

void WaveReconV8::forward(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const {
  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);


    const std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
     std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
     std::shared_ptr<float2D> fm = _fatMask->_mat;
      std::shared_ptr<float2D> s =
       ((std::dynamic_pointer_cast<float2DReg>(_slsq))->_mat);
     std::shared_ptr<float2D> g = _gamma->_mat;
     std::shared_ptr<float2D> gs = _gammaSq->_mat;

  //boundary condition
  #pragma omp parallel for collapse(2)
  for (int ix = FAT; ix < n2-FAT; ix++) { //x
    for (int iz = FAT; iz < n1-FAT; iz++) { //z
      (*d)[0][ix][iz] += //second time deriv
		//C0t_2*((*fm)[ix][iz] * (*m)[1][ix ][iz ])*(*s)[ix][iz] -
       		(C0t_2*(*fm)[ix][iz]* (*m)[1][ix ][iz ] + \
                C1t_2*((*fm)[ix][iz] * (*m)[1-1][ix ][iz ]+(*m)[1+1][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*m)[1][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*m)[1][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[1][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*m)[1][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[1][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*m)[1][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[1][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*m)[1][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[1][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*m)[1][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[1][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*m)[1][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*m)[1][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[1][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*m)[1][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[1][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*m)[1][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[1][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*m)[1][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[1][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*m)[1][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[1][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*m)[2][ix ][iz ]-(*m)[1][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*m)[1][ix ][iz ];
      (*d)[1][ix][iz] += //second time deriv
       		//C0t_4*((*fm)[ix][iz] * (*m)[2][ix ][iz ])*(*s)[ix][iz]  -
       		(C0t_4*(*fm)[ix][iz]* (*m)[2][ix ][iz ] + \
                C1t_4*((*fm)[ix][iz] * (*m)[2-1][ix ][iz ]+(*m)[2 + 1][ix ][iz]) + \
                C2t_4*((*fm)[ix][iz] * (*m)[2-2][ix ][iz ]+(*m)[2 + 2][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*m)[2][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*m)[2][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[2][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*m)[2][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[2][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*m)[2][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[2][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*m)[2][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[2][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*m)[2][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[2][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*m)[2][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*m)[2][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[2][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*m)[2][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[2][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*m)[2][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[2][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*m)[2][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[2][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*m)[2][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[2][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*m)[3][ix ][iz ]-(*m)[2][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*m)[2][ix ][iz ];
      (*d)[2][ix][iz] += //second time deriv
		//C0t_6*((*fm)[ix][iz] * (*m)[3][ix ][iz ])*(*s)[ix][iz]  -
       		(C0t_6*(*fm)[ix][iz]* (*m)[3][ix ][iz ] + \
                C1t_6*((*fm)[ix][iz] * (*m)[3-1][ix ][iz ]+(*m)[3 + 1][ix ][iz]) + \
                C2t_6*((*fm)[ix][iz] * (*m)[3-2][ix ][iz ]+(*m)[3 + 2][ix ][iz]) + \
                C3t_6*((*fm)[ix][iz] * (*m)[3-3][ix ][iz ]+(*m)[3 + 3][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*m)[3][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*m)[3][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[3][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*m)[3][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[3][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*m)[3][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[3][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*m)[3][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[3][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*m)[3][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[3][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*m)[3][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*m)[3][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[3][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*m)[3][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[3][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*m)[3][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[3][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*m)[3][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[3][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*m)[3][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[3][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*m)[4][ix ][iz ]-(*m)[3][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*m)[3][ix ][iz ];
      (*d)[3][ix][iz] += //second time deriv
		//C0t_8*((*fm)[ix][iz] * (*m)[4][ix ][iz ])*(*s)[ix][iz]  -
       		(C0t_8*(*fm)[ix][iz]* (*m)[4][ix ][iz ] + \
                C1t_8*((*fm)[ix][iz] * (*m)[4-1][ix ][iz ]+(*m)[4 + 1][ix ][iz]) + \
                C2t_8*((*fm)[ix][iz] * (*m)[4-2][ix ][iz ]+(*m)[4 + 2][ix ][iz]) + \
                C3t_8*((*fm)[ix][iz] * (*m)[4-3][ix ][iz ]+(*m)[4 + 3][ix ][iz])  + \
                C4t_8*((*fm)[ix][iz] * (*m)[4-4][ix ][iz ]+(*m)[4 + 4][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*m)[4][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*m)[4][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[4][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*m)[4][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[4][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*m)[4][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[4][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*m)[4][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[4][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*m)[4][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[4][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*m)[4][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*m)[4][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[4][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*m)[4][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[4][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*m)[4][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[4][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*m)[4][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[4][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*m)[4][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[4][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*m)[5][ix ][iz ]-(*m)[4][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*m)[4][ix ][iz ];
      (*d)[n3-6][ix][iz] += //second time deriv
		//C0t_8*((*fm)[ix][iz] * (*m)[n3-5][ix ][iz ])*(*s)[ix][iz]  -
       		(C0t_8*(*fm)[ix][iz]* (*m)[n3-5][ix ][iz ]+ \
                C1t_8*((*fm)[ix][iz] * (*m)[n3-5-1][ix ][iz ]+(*m)[n3-5 + 1][ix ][iz]) + \
                C2t_8*((*fm)[ix][iz] * (*m)[n3-5-2][ix ][iz ]+(*m)[n3-5 + 2][ix ][iz]) + \
                C3t_8*((*fm)[ix][iz] * (*m)[n3-5-3][ix ][iz ]+(*m)[n3-5 + 3][ix ][iz])  + \
                C4t_8*((*fm)[ix][iz] * (*m)[n3-5-4][ix ][iz ]+(*m)[n3-5 + 4][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*m)[n3-5][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*m)[n3-5][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[n3-5][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*m)[n3-5][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[n3-5][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*m)[n3-5][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[n3-5][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*m)[n3-5][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[n3-5][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*m)[n3-5][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[n3-5][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*m)[n3-5][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*m)[n3-5][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[n3-5][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*m)[n3-5][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[n3-5][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*m)[n3-5][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[n3-5][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*m)[n3-5][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[n3-5][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*m)[n3-5][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[n3-5][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*m)[n3-4][ix ][iz ]-(*m)[n3-5][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*m)[n3-5][ix ][iz ];
      (*d)[n3-5][ix][iz] += //second time deriv
		//C0t_6*((*fm)[ix][iz] * (*m)[n3-4][ix][iz])*(*s)[ix][iz]  -
       		(C0t_6*(*fm)[ix][iz]* (*m)[n3-4][ix ][iz ] + \
                C1t_6*((*fm)[ix][iz] * (*m)[n3-4-1][ix ][iz ]+(*m)[n3-4 + 1][ix ][iz]) + \
                C2t_6*((*fm)[ix][iz] * (*m)[n3-4-2][ix ][iz ]+(*m)[n3-4 + 2][ix ][iz]) + \
                C3t_6*((*fm)[ix][iz] * (*m)[n3-4-3][ix ][iz ]+(*m)[n3-4 + 3][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*m)[n3-4][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*m)[n3-4][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[n3-4][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*m)[n3-4][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[n3-4][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*m)[n3-4][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[n3-4][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*m)[n3-4][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[n3-4][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*m)[n3-4][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[n3-4][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*m)[n3-4][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*m)[n3-4][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[n3-4][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*m)[n3-4][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[n3-4][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*m)[n3-4][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[n3-4][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*m)[n3-4][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[n3-4][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*m)[n3-4][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[n3-4][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*m)[n3-3][ix ][iz ]-(*m)[n3-4][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*m)[n3-4][ix ][iz ];
      (*d)[n3-4][ix][iz] += //second time deriv
		//C0t_4*((*fm)[ix][iz] * (*m)[n3-3][ix ][iz ])*(*s)[ix][iz]  -
       		(C0t_4*(*fm)[ix][iz]* (*m)[n3-3][ix ][iz ] + \
                C1t_4*((*fm)[ix][iz] * (*m)[n3-3-1][ix ][iz ]+(*m)[n3-3 + 1][ix ][iz]) + \
                C2t_4*((*fm)[ix][iz] * (*m)[n3-3-2][ix ][iz ]+(*m)[n3-3 + 2][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*m)[n3-3][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*m)[n3-3][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[n3-3][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*m)[n3-3][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[n3-3][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*m)[n3-3][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[n3-3][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*m)[n3-3][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[n3-3][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*m)[n3-3][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[n3-3][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*m)[n3-3][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*m)[n3-3][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[n3-3][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*m)[n3-3][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[n3-3][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*m)[n3-3][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[n3-3][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*m)[n3-3][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[n3-3][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*m)[n3-3][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[n3-3][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*m)[n3-2][ix ][iz ]-(*m)[n3-3][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*m)[n3-3][ix ][iz ];
      (*d)[n3-3][ix][iz] += //second time deriv
		//C0t_2*((*fm)[ix][iz] * (*m)[n3-2][ix ][iz ])*(*s)[ix][iz]  -
       		(C0t_2*(*fm)[ix][iz]* (*m)[n3-2][ix ][iz ] + \
                C1t_2*((*fm)[ix][iz] * (*m)[n3-2-1][ix ][iz ]+(*m)[n3-2 + 1][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*m)[n3-2][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*m)[n3-2][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[n3-2][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*m)[n3-2][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[n3-2][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*m)[n3-2][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[n3-2][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*m)[n3-2][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[n3-2][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*m)[n3-2][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[n3-2][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*m)[n3-2][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*m)[n3-2][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[n3-2][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*m)[n3-2][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[n3-2][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*m)[n3-2][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[n3-2][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*m)[n3-2][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[n3-2][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*m)[n3-2][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[n3-2][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*m)[n3-1][ix ][iz ]-(*m)[n3-2][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*m)[n3-2][ix ][iz ];
      (*d)[n3-2][ix][iz] += (*m)[0 ][ix ][iz]-(*m)[1 ][ix ][iz];
      (*d)[n3-1][ix][iz] += (*m)[1 ][ix ][iz];
    }
  }
  #pragma omp parallel for collapse(3)
  for (int it = 5; it < n3-5; it++) { //time
    for (int ix = FAT; ix < n2-FAT; ix++) { //x
      for (int iz = FAT; iz < n1-FAT; iz++) { //z
        (*d)[it-1][ix][iz] +=//second time deriv
		//C0t_10*((*fm)[ix][iz] * (*m)[it][ix ][iz ])*(*s)[ix][iz]  -
       		(C0t_10*(*fm)[ix][iz]* (*m)[it][ix ][iz ] + \
                C1t_10*((*fm)[ix][iz] * (*m)[it-1][ix ][iz ]+(*m)[it + 1][ix ][iz]) + \
                C2t_10*((*fm)[ix][iz] * (*m)[it-2][ix ][iz ]+(*m)[it + 2][ix ][iz]) + \
                C3t_10*((*fm)[ix][iz] * (*m)[it-3][ix ][iz ]+(*m)[it + 3][ix ][iz]) + \
                C4t_10*((*fm)[ix][iz] * (*m)[it-4][ix ][iz ]+(*m)[it + 4][ix ][iz]) + \
                C5t_10*((*fm)[ix][iz] * (*m)[it-5][ix ][iz ]+(*m)[it + 5][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*m)[it][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*m)[it][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[it][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*m)[it][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[it][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*m)[it][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[it][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*m)[it][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[it][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*m)[it][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[it][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*m)[it][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*m)[it][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[it][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*m)[it][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[it][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*m)[it][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[it][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*m)[it][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[it][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*m)[it][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[it][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*m)[it+1][ix ][iz ]-(*m)[it][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*m)[it][ix ][iz ];
      }
    }
  }

}

void WaveReconV8::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

    std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
    const std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
      std::shared_ptr<float2D> s =
       ((std::dynamic_pointer_cast<float2DReg>(_slsq))->_mat);
     std::shared_ptr<float2D> fm = _fatMask->_mat;
     std::shared_ptr<float2D> g = _gamma->_mat;
     std::shared_ptr<float2D> gs = _gammaSq->_mat;


  #pragma omp parallel for collapse(2)
  for (int ix = FAT; ix < n2-FAT; ix++) { //x
    for (int iz = FAT; iz < n1-FAT; iz++) { //z
      (*m)[0][ix][iz] +=   //second time deriv
                         (C1t_2*((*fm)[ix][iz]*(*d)[0][ix ][iz]) + \
                         C2t_4*((*fm)[ix][iz]*(*d)[1][ix ][iz]) + \
                         C3t_6*((*fm)[ix][iz]*(*d)[2][ix ][iz]) + \
                         C4t_8*((*fm)[ix][iz]*(*d)[3][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[4][ix ][iz]))*(*s)[ix][iz] + \
                         //initial condition
       			 (*d)[n3-2][ix ][iz ];
      (*m)[1][ix][iz] +=   //second time deriv
			//C0t_2*((*fm)[ix][iz]*(*d)[0][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_2*((*fm)[ix][iz]*(*d)[0][ix ][iz]) + \
                         C1t_4*((*fm)[ix][iz]*(*d)[1][ix ][iz]) + \
                         C2t_6*((*fm)[ix][iz]*(*d)[2][ix ][iz]) + \
                         C3t_8*((*fm)[ix][iz]*(*d)[3][ix ][iz]) + \
                         C4t_10*((*fm)[ix][iz]*(*d)[4][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[0][ix ][iz ] + \
                         C1x * ((*d)[0][ix + 1 ][iz ] + (*d)[0][ix - 1 ][iz ]) + \
                         C2x * ((*d)[0][ix + 2 ][iz ] + (*d)[0][ix - 2 ][iz ]) + \
                         C3x * ((*d)[0][ix + 3 ][iz ] + (*d)[0][ix - 3 ][iz ]) + \
                         C4x * ((*d)[0][ix + 4 ][iz ] + (*d)[0][ix - 4 ][iz ]) + \
                         C5x * ((*d)[0][ix + 5 ][iz ] + (*d)[0][ix - 5 ][iz ]) + \
                         C0z * ((*d)[0][ix ][iz ]) + \
                         C1z * ((*d)[0][ix ][iz + 1 ] + (*d)[0][ix ][iz - 1 ]) + \
                         C2z * ((*d)[0][ix ][iz + 2 ] + (*d)[0][ix ][iz - 2 ]) + \
                         C3z * ((*d)[0][ix ][iz + 3 ] + (*d)[0][ix ][iz - 3 ]) + \
                         C4z * ((*d)[0][ix ][iz + 4 ] + (*d)[0][ix ][iz - 4 ]) + \
                         C5z * ((*d)[0][ix ][iz + 5 ] + (*d)[0][ix ][iz - 5 ]))*(*fm)[ix][iz] +
                         //initial condition
       			 (*d)[n3-1][ix ][iz ] - (*d)[n3-2][ix ][iz ]+ \
		         //sponge first term
		         (*g)[ix][iz]*(-1)*(*d)[0][ix ][iz ] + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[0][ix ][iz ];
      (*m)[2][ix][iz] +=   //second time deriv
			//C0t_4*((*fm)[ix][iz]*(*d)[1][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_4*((*fm)[ix][iz]*(*d)[1][ix ][iz]) + \
                         C1t_2*(*d)[1-1][ix ][iz] + C1t_6*((*fm)[ix][iz]*(*d)[1+1][ix ][iz]) + \
                         C2t_8*((*fm)[ix][iz]*(*d)[1+2][ix ][iz]) + \
                         C3t_10*((*fm)[ix][iz]*(*d)[1+3][ix ][iz]) + \
                         C4t_10*((*fm)[ix][iz]*(*d)[1+4][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[1+5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[1][ix ][iz ] + \
                         C1x * ((*d)[1][ix + 1 ][iz ] + (*d)[1][ix - 1 ][iz ]) + \
                         C2x * ((*d)[1][ix + 2 ][iz ] + (*d)[1][ix - 2 ][iz ]) + \
                         C3x * ((*d)[1][ix + 3 ][iz ] + (*d)[1][ix - 3 ][iz ]) + \
                         C4x * ((*d)[1][ix + 4 ][iz ] + (*d)[1][ix - 4 ][iz ]) + \
                         C5x * ((*d)[1][ix + 5 ][iz ] + (*d)[1][ix - 5 ][iz ]) + \
                         C0z * ((*d)[1][ix ][iz ]) + \
                         C1z * ((*d)[1][ix ][iz + 1 ] + (*d)[1][ix ][iz - 1 ]) + \
                         C2z * ((*d)[1][ix ][iz + 2 ] + (*d)[1][ix ][iz - 2 ]) + \
                         C3z * ((*d)[1][ix ][iz + 3 ] + (*d)[1][ix ][iz - 3 ]) + \
                         C4z * ((*d)[1][ix ][iz + 4 ] + (*d)[1][ix ][iz - 4 ]) + \
                         C5z * ((*d)[1][ix ][iz + 5 ] + (*d)[1][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[0][ix ][iz ]-(*d)[1][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[1][ix ][iz ];
      (*m)[3][ix][iz] +=   //second time deriv
			//C0t_6*((*fm)[ix][iz]*(*d)[2][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_6*((*fm)[ix][iz]*(*d)[2][ix ][iz]) + \
                         C1t_4*(*d)[1][ix ][iz] + C1t_8*((*fm)[ix][iz]*(*d)[3][ix ][iz]) + \
                         C2t_10*((*fm)[ix][iz]*(*d)[4][ix ][iz]) + \
                         C3t_10*((*fm)[ix][iz]*(*d)[5][ix ][iz]) + \
                         C4t_10*((*fm)[ix][iz]*(*d)[6][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[7][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[2][ix ][iz ] + \
                         C1x * ((*d)[2][ix + 1 ][iz ] + (*d)[2][ix - 1 ][iz ]) + \
                         C2x * ((*d)[2][ix + 2 ][iz ] + (*d)[2][ix - 2 ][iz ]) + \
                         C3x * ((*d)[2][ix + 3 ][iz ] + (*d)[2][ix - 3 ][iz ]) + \
                         C4x * ((*d)[2][ix + 4 ][iz ] + (*d)[2][ix - 4 ][iz ]) + \
                         C5x * ((*d)[2][ix + 5 ][iz ] + (*d)[2][ix - 5 ][iz ]) + \
                         C0z * ((*d)[2][ix ][iz ]) + \
                         C1z * ((*d)[2][ix ][iz + 1 ] + (*d)[2][ix ][iz - 1 ]) + \
                         C2z * ((*d)[2][ix ][iz + 2 ] + (*d)[2][ix ][iz - 2 ]) + \
                         C3z * ((*d)[2][ix ][iz + 3 ] + (*d)[2][ix ][iz - 3 ]) + \
                         C4z * ((*d)[2][ix ][iz + 4 ] + (*d)[2][ix ][iz - 4 ]) + \
                         C5z * ((*d)[2][ix ][iz + 5 ] + (*d)[2][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[1][ix ][iz ]-(*d)[2][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[2][ix ][iz ];
      (*m)[4][ix][iz] +=   //second time deriv
                         //C0t_8*((*fm)[ix][iz]*(*d)[3][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_8*((*fm)[ix][iz]*(*d)[3][ix ][iz]) + \
                         C1t_6*(*d)[2][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[4][ix ][iz]) + \
                         C2t_4*(*d)[1][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[5][ix ][iz]) + \
                         C3t_10*((*fm)[ix][iz]*(*d)[6][ix ][iz]) + \
                         C4t_10*((*fm)[ix][iz]*(*d)[7][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[8][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[3][ix ][iz ] + \
                         C1x * ((*d)[3][ix + 1 ][iz ] + (*d)[3][ix - 1 ][iz ]) + \
                         C2x * ((*d)[3][ix + 2 ][iz ] + (*d)[3][ix - 2 ][iz ]) + \
                         C3x * ((*d)[3][ix + 3 ][iz ] + (*d)[3][ix - 3 ][iz ]) + \
                         C4x * ((*d)[3][ix + 4 ][iz ] + (*d)[3][ix - 4 ][iz ]) + \
                         C5x * ((*d)[3][ix + 5 ][iz ] + (*d)[3][ix - 5 ][iz ]) + \
                         C0z * ((*d)[3][ix ][iz ]) + \
                         C1z * ((*d)[3][ix ][iz + 1 ] + (*d)[3][ix ][iz - 1 ]) + \
                         C2z * ((*d)[3][ix ][iz + 2 ] + (*d)[3][ix ][iz - 2 ]) + \
                         C3z * ((*d)[3][ix ][iz + 3 ] + (*d)[3][ix ][iz - 3 ]) + \
                         C4z * ((*d)[3][ix ][iz + 4 ] + (*d)[3][ix ][iz - 4 ]) + \
                         C5z * ((*d)[3][ix ][iz + 5 ] + (*d)[3][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[2][ix ][iz ]-(*d)[3][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[3][ix ][iz ];
      (*m)[5][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[4][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_10*((*fm)[ix][iz]*(*d)[4][ix ][iz]) + \
                         C1t_8*(*d)[3][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[5][ix ][iz]) + \
                         C2t_6*(*d)[2][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[6][ix ][iz]) + \
                         C3t_10*((*fm)[ix][iz]*(*d)[7][ix ][iz]) + \
                         C4t_10*((*fm)[ix][iz]*(*d)[8][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[9][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[4][ix ][iz ] + \
                         C1x * ((*d)[4][ix + 1 ][iz ] + (*d)[4][ix - 1 ][iz ]) + \
                         C2x * ((*d)[4][ix + 2 ][iz ] + (*d)[4][ix - 2 ][iz ]) + \
                         C3x * ((*d)[4][ix + 3 ][iz ] + (*d)[4][ix - 3 ][iz ]) + \
                         C4x * ((*d)[4][ix + 4 ][iz ] + (*d)[4][ix - 4 ][iz ]) + \
                         C5x * ((*d)[4][ix + 5 ][iz ] + (*d)[4][ix - 5 ][iz ]) + \
                         C0z * ((*d)[4][ix ][iz ]) + \
                         C1z * ((*d)[4][ix ][iz + 1 ] + (*d)[4][ix ][iz - 1 ]) + \
                         C2z * ((*d)[4][ix ][iz + 2 ] + (*d)[4][ix ][iz - 2 ]) + \
                         C3z * ((*d)[4][ix ][iz + 3 ] + (*d)[4][ix ][iz - 3 ]) + \
                         C4z * ((*d)[4][ix ][iz + 4 ] + (*d)[4][ix ][iz - 4 ]) + \
                         C5z * ((*d)[4][ix ][iz + 5 ] + (*d)[4][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[3][ix ][iz ]-(*d)[4][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[4][ix ][iz ];
      (*m)[6][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[5][ix ][iz])*(*s)[ix][iz]  -
                        (C0t_10*((*fm)[ix][iz]*(*d)[5][ix ][iz]) + \
                         C1t_10*(*d)[4][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[6][ix ][iz]) + \
                         C2t_8*(*d)[3][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[7][ix ][iz]) + \
                         C3t_6*(*d)[2][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[8][ix ][iz]) + \
                         C4t_10*((*fm)[ix][iz]*(*d)[9][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[10][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[5][ix ][iz ] + \
                         C1x * ((*d)[5][ix + 1 ][iz ] + (*d)[5][ix - 1 ][iz ]) + \
                         C2x * ((*d)[5][ix + 2 ][iz ] + (*d)[5][ix - 2 ][iz ]) + \
                         C3x * ((*d)[5][ix + 3 ][iz ] + (*d)[5][ix - 3 ][iz ]) + \
                         C4x * ((*d)[5][ix + 4 ][iz ] + (*d)[5][ix - 4 ][iz ]) + \
                         C5x * ((*d)[5][ix + 5 ][iz ] + (*d)[5][ix - 5 ][iz ]) + \
                         C0z * ((*d)[5][ix ][iz ]) + \
                         C1z * ((*d)[5][ix ][iz + 1 ] + (*d)[5][ix ][iz - 1 ]) + \
                         C2z * ((*d)[5][ix ][iz + 2 ] + (*d)[5][ix ][iz - 2 ]) + \
                         C3z * ((*d)[5][ix ][iz + 3 ] + (*d)[5][ix ][iz - 3 ]) + \
                         C4z * ((*d)[5][ix ][iz + 4 ] + (*d)[5][ix ][iz - 4 ]) + \
                         C5z * ((*d)[5][ix ][iz + 5 ] + (*d)[5][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[4][ix ][iz ]-(*d)[5][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[5][ix ][iz ];
      (*m)[7][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[6][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_10*((*fm)[ix][iz]*(*d)[6][ix ][iz])+ \
                         C1t_10*(*d)[5][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[7][ix ][iz]) + \
                         C2t_10*(*d)[4][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[8][ix ][iz]) + \
                         C3t_8*(*d)[3][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[9][ix ][iz]) + \
                         C4t_10*((*fm)[ix][iz]*(*d)[10][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[11][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[6][ix ][iz ] + \
                         C1x * ((*d)[6][ix + 1 ][iz ] + (*d)[6][ix - 1 ][iz ]) + \
                         C2x * ((*d)[6][ix + 2 ][iz ] + (*d)[6][ix - 2 ][iz ]) + \
                         C3x * ((*d)[6][ix + 3 ][iz ] + (*d)[6][ix - 3 ][iz ]) + \
                         C4x * ((*d)[6][ix + 4 ][iz ] + (*d)[6][ix - 4 ][iz ]) + \
                         C5x * ((*d)[6][ix + 5 ][iz ] + (*d)[6][ix - 5 ][iz ]) + \
                         C0z * ((*d)[6][ix ][iz ]) + \
                         C1z * ((*d)[6][ix ][iz + 1 ] + (*d)[6][ix ][iz - 1 ]) + \
                         C2z * ((*d)[6][ix ][iz + 2 ] + (*d)[6][ix ][iz - 2 ]) + \
                         C3z * ((*d)[6][ix ][iz + 3 ] + (*d)[6][ix ][iz - 3 ]) + \
                         C4z * ((*d)[6][ix ][iz + 4 ] + (*d)[6][ix ][iz - 4 ]) + \
                         C5z * ((*d)[6][ix ][iz + 5 ] + (*d)[6][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[5][ix ][iz ]-(*d)[6][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[6][ix ][iz ];
      (*m)[8][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[7][ix ][iz])*(*s)[ix][iz] -
                        (C0t_10*((*fm)[ix][iz]*(*d)[7][ix ][iz]) + \
                         C1t_10*(*d)[6][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[8][ix ][iz]) + \
                         C2t_10*(*d)[5][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[9][ix ][iz]) + \
                         C3t_10*(*d)[4][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[10][ix ][iz]) + \
                         C4t_8*(*d)[3][ix ][iz] + C4t_10*((*fm)[ix][iz]*(*d)[11][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[12][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[7][ix ][iz ]  + \
                         C1x * ((*d)[7][ix + 1 ][iz ] + (*d)[7][ix - 1 ][iz ]) + \
                         C2x * ((*d)[7][ix + 2 ][iz ] + (*d)[7][ix - 2 ][iz ]) + \
                         C3x * ((*d)[7][ix + 3 ][iz ] + (*d)[7][ix - 3 ][iz ]) + \
                         C4x * ((*d)[7][ix + 4 ][iz ] + (*d)[7][ix - 4 ][iz ]) + \
                         C5x * ((*d)[7][ix + 5 ][iz ] + (*d)[7][ix - 5 ][iz ]) + \
                         C0z * ((*d)[7][ix ][iz ]) + \
                         C1z * ((*d)[7][ix ][iz + 1 ] + (*d)[7][ix ][iz - 1 ]) + \
                         C2z * ((*d)[7][ix ][iz + 2 ] + (*d)[7][ix ][iz - 2 ]) + \
                         C3z * ((*d)[7][ix ][iz + 3 ] + (*d)[7][ix ][iz - 3 ]) + \
                         C4z * ((*d)[7][ix ][iz + 4 ] + (*d)[7][ix ][iz - 4 ]) + \
                         C5z * ((*d)[7][ix ][iz + 5 ] + (*d)[7][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[6][ix ][iz ]-(*d)[7][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[7][ix ][iz ];
      (*m)[9][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[8][ix ][iz])*(*s)[ix][iz] -
                        (C0t_10*((*fm)[ix][iz]*(*d)[8][ix ][iz]) + \
                         C1t_10*(*d)[7][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[9][ix ][iz]) + \
                         C2t_10*(*d)[6][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[10][ix ][iz]) + \
                         C3t_10*(*d)[5][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[11][ix ][iz]) + \
                         C4t_10*(*d)[4][ix ][iz] + C4t_10*((*fm)[ix][iz]*(*d)[12][ix ][iz]) + \
                         C5t_10*((*fm)[ix][iz]*(*d)[13][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[8][ix ][iz ] + \
                         C1x * ((*d)[8][ix + 1 ][iz ] + (*d)[8][ix - 1 ][iz ]) + \
                         C2x * ((*d)[8][ix + 2 ][iz ] + (*d)[8][ix - 2 ][iz ]) + \
                         C3x * ((*d)[8][ix + 3 ][iz ] + (*d)[8][ix - 3 ][iz ]) + \
                         C4x * ((*d)[8][ix + 4 ][iz ] + (*d)[8][ix - 4 ][iz ]) + \
                         C5x * ((*d)[8][ix + 5 ][iz ] + (*d)[8][ix - 5 ][iz ]) + \
                         C0z * ((*d)[8][ix ][iz ]) + \
                         C1z * ((*d)[8][ix ][iz + 1 ] + (*d)[8][ix ][iz - 1 ]) + \
                         C2z * ((*d)[8][ix ][iz + 2 ] + (*d)[8][ix ][iz - 2 ]) + \
                         C3z * ((*d)[8][ix ][iz + 3 ] + (*d)[8][ix ][iz - 3 ]) + \
                         C4z * ((*d)[8][ix ][iz + 4 ] + (*d)[8][ix ][iz - 4 ]) + \
                         C5z * ((*d)[8][ix ][iz + 5 ] + (*d)[8][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[7][ix ][iz ]-(*d)[8][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[8][ix ][iz ];
      (*m)[n3-10][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[n3-11][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_10*((*fm)[ix][iz]*(*d)[n3-11][ix ][iz]) + \
                         C1t_10*(*d)[n3-11-1][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[n3-11+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-11-2][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[n3-11+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-11-3][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[n3-11+3][ix ][iz]) + \
                         C4t_10*(*d)[n3-11-4][ix ][iz] + C4t_10*((*fm)[ix][iz]*(*d)[n3-11+4][ix ][iz]) + \
                         C5t_10*((*d)[n3-11-5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-11][ix ][iz ] + \
                         C1x * ((*d)[n3-11][ix + 1 ][iz ] + (*d)[n3-11][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-11][ix + 2 ][iz ] + (*d)[n3-11][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-11][ix + 3 ][iz ] + (*d)[n3-11][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-11][ix + 4 ][iz ] + (*d)[n3-11][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-11][ix + 5 ][iz ] + (*d)[n3-11][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-11][ix ][iz ]) + \
                         C1z * ((*d)[n3-11][ix ][iz + 1 ] + (*d)[n3-11][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-11][ix ][iz + 2 ] + (*d)[n3-11][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-11][ix ][iz + 3 ] + (*d)[n3-11][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-11][ix ][iz + 4 ] + (*d)[n3-11][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-11][ix ][iz + 5 ] + (*d)[n3-11][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-12][ix ][iz ]-(*d)[n3-11][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-11][ix ][iz ];
      (*m)[n3-9][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[n3-10][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_10*((*fm)[ix][iz]*(*d)[n3-10][ix ][iz]) + \
                         C1t_10*(*d)[n3-10-1][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[n3-10+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-10-2][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[n3-10+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-10-3][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[n3-10+3][ix ][iz]) + \
                         C4t_10*(*d)[n3-10-4][ix ][iz] + C4t_8*((*fm)[ix][iz]*(*d)[n3-10+4][ix ][iz]) + \
                         C5t_10*((*d)[n3-10-5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-10][ix ][iz ] + \
                         C1x * ((*d)[n3-10][ix + 1 ][iz ] + (*d)[n3-10][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-10][ix + 2 ][iz ] + (*d)[n3-10][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-10][ix + 3 ][iz ] + (*d)[n3-10][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-10][ix + 4 ][iz ] + (*d)[n3-10][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-10][ix + 5 ][iz ] + (*d)[n3-10][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-10][ix ][iz ]) + \
                         C1z * ((*d)[n3-10][ix ][iz + 1 ] + (*d)[n3-10][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-10][ix ][iz + 2 ] + (*d)[n3-10][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-10][ix ][iz + 3 ] + (*d)[n3-10][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-10][ix ][iz + 4 ] + (*d)[n3-10][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-10][ix ][iz + 5 ] + (*d)[n3-10][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-11][ix ][iz ]-(*d)[n3-10][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-10][ix ][iz ];
      (*m)[n3-8][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[n3-9][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_10*((*fm)[ix][iz]*(*d)[n3-9][ix ][iz]) + \
                         C1t_10*(*d)[n3-9-1][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[n3-9+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-9-2][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[n3-9+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-9-3][ix ][iz] + C3t_8*((*fm)[ix][iz]*(*d)[n3-9+3][ix ][iz]) + \
                         C4t_10*(*d)[n3-9-4][ix ][iz] + \
                         C5t_10*((*d)[n3-9-5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-9][ix ][iz ] + \
                         C1x * ((*d)[n3-9][ix + 1 ][iz ] + (*d)[n3-9][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-9][ix + 2 ][iz ] + (*d)[n3-9][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-9][ix + 3 ][iz ] + (*d)[n3-9][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-9][ix + 4 ][iz ] + (*d)[n3-9][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-9][ix + 5 ][iz ] + (*d)[n3-9][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-9][ix ][iz ]) + \
                         C1z * ((*d)[n3-9][ix ][iz + 1 ] + (*d)[n3-9][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-9][ix ][iz + 2 ] + (*d)[n3-9][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-9][ix ][iz + 3 ] + (*d)[n3-9][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-9][ix ][iz + 4 ] + (*d)[n3-9][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-9][ix ][iz + 5 ] + (*d)[n3-9][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-10][ix ][iz ]-(*d)[n3-9][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-9][ix ][iz ];
      (*m)[n3-7][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[n3-8][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_10*((*fm)[ix][iz]*(*d)[n3-8][ix ][iz]) + \
                         C1t_10*(*d)[n3-8-1][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[n3-8+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-8-2][ix ][iz] + C2t_8*((*fm)[ix][iz]*(*d)[n3-8+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-8-3][ix ][iz] + C3t_6*((*fm)[ix][iz]*(*d)[n3-8+3][ix ][iz]) + \
                         C4t_10*(*d)[n3-8-4][ix ][iz] + \
                         C5t_10*((*d)[n3-8-5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-8][ix ][iz ]  + \
                         C1x * ((*d)[n3-8][ix + 1 ][iz ] + (*d)[n3-8][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-8][ix + 2 ][iz ] + (*d)[n3-8][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-8][ix + 3 ][iz ] + (*d)[n3-8][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-8][ix + 4 ][iz ] + (*d)[n3-8][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-8][ix + 5 ][iz ] + (*d)[n3-8][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-8][ix ][iz ]) + \
                         C1z * ((*d)[n3-8][ix ][iz + 1 ] + (*d)[n3-8][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-8][ix ][iz + 2 ] + (*d)[n3-8][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-8][ix ][iz + 3 ] + (*d)[n3-8][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-8][ix ][iz + 4 ] + (*d)[n3-8][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-8][ix ][iz + 5 ] + (*d)[n3-8][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-9][ix ][iz ]-(*d)[n3-8][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-8][ix ][iz ];
      (*m)[n3-6][ix][iz] +=   //second time deriv
			//C0t_10*((*fm)[ix][iz]*(*d)[n3-7][ix ][iz])*(*s)[ix][iz] -
                         (C0t_10*((*fm)[ix][iz]*(*d)[n3-7][ix ][iz]) + \
                         C1t_10*(*d)[n3-7-1][ix ][iz] + C1t_8*((*fm)[ix][iz]*(*d)[n3-7+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-7-2][ix ][iz] + C2t_6*((*fm)[ix][iz]*(*d)[n3-7+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-7-3][ix ][iz] + \
                         C4t_10*(*d)[n3-7-4][ix ][iz] + \
                         C5t_10*((*d)[n3-7-5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-7][ix ][iz ]  + \
                         C1x * ((*d)[n3-7][ix + 1 ][iz ] + (*d)[n3-7][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-7][ix + 2 ][iz ] + (*d)[n3-7][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-7][ix + 3 ][iz ] + (*d)[n3-7][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-7][ix + 4 ][iz ] + (*d)[n3-7][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-7][ix + 5 ][iz ] + (*d)[n3-7][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-7][ix ][iz ]) + \
                         C1z * ((*d)[n3-7][ix ][iz + 1 ] + (*d)[n3-7][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-7][ix ][iz + 2 ] + (*d)[n3-7][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-7][ix ][iz + 3 ] + (*d)[n3-7][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-7][ix ][iz + 4 ] + (*d)[n3-7][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-7][ix ][iz + 5 ] + (*d)[n3-7][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-8][ix ][iz ]-(*d)[n3-7][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-7][ix ][iz ];
      (*m)[n3-5][ix][iz] +=   //second time deriv
			//C0t_8*((*fm)[ix][iz]*(*d)[n3-6][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_8*((*fm)[ix][iz]*(*d)[n3-6][ix ][iz]) + \
                         C1t_10*(*d)[n3-6-1][ix ][iz] + C1t_6*((*fm)[ix][iz]*(*d)[n3-6+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-6-2][ix ][iz] + C2t_4*((*fm)[ix][iz]*(*d)[n3-6+2][ix ][iz]) + \
                         C3t_10*(*d)[n3-6-3][ix ][iz] + \
                         C4t_10*(*d)[n3-6-4][ix ][iz] + \
                         C5t_10*((*d)[n3-6-5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-6][ix ][iz ] + \
                         C1x * ((*d)[n3-6][ix + 1 ][iz ] + (*d)[n3-6][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-6][ix + 2 ][iz ] + (*d)[n3-6][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-6][ix + 3 ][iz ] + (*d)[n3-6][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-6][ix + 4 ][iz ] + (*d)[n3-6][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-6][ix + 5 ][iz ] + (*d)[n3-6][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-6][ix ][iz ]) + \
                         C1z * ((*d)[n3-6][ix ][iz + 1 ] + (*d)[n3-6][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-6][ix ][iz + 2 ] + (*d)[n3-6][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-6][ix ][iz + 3 ] + (*d)[n3-6][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-6][ix ][iz + 4 ] + (*d)[n3-6][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-6][ix ][iz + 5 ] + (*d)[n3-6][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-7][ix ][iz ]-(*d)[n3-6][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-6][ix ][iz ];
      (*m)[n3-4][ix][iz] +=   //second time deriv
			//C0t_6*((*fm)[ix][iz]*(*d)[n3-5][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_6*((*fm)[ix][iz]*(*d)[n3-5][ix ][iz]) + \
                         C1t_8*(*d)[n3-5-1][ix ][iz] + C1t_4*((*fm)[ix][iz]*(*d)[n3-5+1][ix ][iz]) + \
                         C2t_10*(*d)[n3-5-2][ix ][iz] + \
                         C3t_10*(*d)[n3-5-3][ix ][iz] + \
                         C4t_10*(*d)[n3-5-4][ix ][iz] + \
                         C5t_10*((*d)[n3-5-5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-5][ix ][iz ] + \
                         C1x * ((*d)[n3-5][ix + 1 ][iz ] + (*d)[n3-5][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-5][ix + 2 ][iz ] + (*d)[n3-5][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-5][ix + 3 ][iz ] + (*d)[n3-5][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-5][ix + 4 ][iz ] + (*d)[n3-5][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-5][ix + 5 ][iz ] + (*d)[n3-5][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-5][ix ][iz ]) + \
                         C1z * ((*d)[n3-5][ix ][iz + 1 ] + (*d)[n3-5][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-5][ix ][iz + 2 ] + (*d)[n3-5][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-5][ix ][iz + 3 ] + (*d)[n3-5][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-5][ix ][iz + 4 ] + (*d)[n3-5][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-5][ix ][iz + 5 ] + (*d)[n3-5][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-6][ix ][iz ]-(*d)[n3-5][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-5][ix ][iz ];
      (*m)[n3-3][ix][iz] +=   //second time deriv
			//C0t_4*((*fm)[ix][iz]*(*d)[n3-4][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_4*((*fm)[ix][iz]*(*d)[n3-4][ix ][iz]) + \
                         C1t_6*(*d)[n3-4-1][ix ][iz] + C1t_2*((*fm)[ix][iz]*(*d)[n3-4+1][ix ][iz]) + \
                         C2t_8*(*d)[n3-4-2][ix ][iz] + \
                         C3t_10*(*d)[n3-4-3][ix ][iz] + \
                         C4t_10*(*d)[n3-4-4][ix ][iz] + \
                         C5t_10*((*d)[n3-4-5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-4][ix ][iz ] + \
                         C1x * ((*d)[n3-4][ix + 1 ][iz ] + (*d)[n3-4][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-4][ix + 2 ][iz ] + (*d)[n3-4][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-4][ix + 3 ][iz ] + (*d)[n3-4][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-4][ix + 4 ][iz ] + (*d)[n3-4][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-4][ix + 5 ][iz ] + (*d)[n3-4][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-4][ix ][iz ]) + \
                         C1z * ((*d)[n3-4][ix ][iz + 1 ] + (*d)[n3-4][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-4][ix ][iz + 2 ] + (*d)[n3-4][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-4][ix ][iz + 3 ] + (*d)[n3-4][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-4][ix ][iz + 4 ] + (*d)[n3-4][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-4][ix ][iz + 5 ] + (*d)[n3-4][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-5][ix ][iz ]-(*d)[n3-4][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-4][ix ][iz ];
      (*m)[n3-2][ix][iz] +=   //second time deriv
			//C0t_2*((*fm)[ix][iz]*(*d)[n3-3][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_2*((*fm)[ix][iz]*(*d)[n3-3][ix ][iz]) + \
                         C1t_4*(*d)[n3-3-1][ix ][iz] + \
                         C2t_6*(*d)[n3-3-2][ix ][iz] + \
                         C3t_8*(*d)[n3-3-3][ix ][iz] + \
                         C4t_10*(*d)[n3-3-4][ix ][iz] + \
                         C5t_10*((*d)[n3-3-5][ix ][iz]))*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-3][ix ][iz ] + \
                         C1x * ((*d)[n3-3][ix + 1 ][iz ] + (*d)[n3-3][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-3][ix + 2 ][iz ] + (*d)[n3-3][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-3][ix + 3 ][iz ] + (*d)[n3-3][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-3][ix + 4 ][iz ] + (*d)[n3-3][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-3][ix + 5 ][iz ] + (*d)[n3-3][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-3][ix ][iz ]) + \
                         C1z * ((*d)[n3-3][ix ][iz + 1 ] + (*d)[n3-3][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-3][ix ][iz + 2 ] + (*d)[n3-3][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-3][ix ][iz + 3 ] + (*d)[n3-3][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-3][ix ][iz + 4 ] + (*d)[n3-3][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-3][ix ][iz + 5 ] + (*d)[n3-3][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-4][ix ][iz ]-(*d)[n3-3][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-3][ix ][iz ];
      (*m)[n3-1][ix][iz] +=   //second time deriv
                         (C1t_2*(*d)[n3-2-1][ix ][iz] + \
                         C2t_4*(*d)[n3-2-2][ix ][iz] + \
                         C3t_6*(*d)[n3-2-3][ix ][iz] + \
                         C4t_8*(*d)[n3-2-4][ix ][iz] + \
                         C5t_10*((*d)[n3-2-5][ix ][iz]))*(*s)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*(*d)[n3-3][ix ][iz ];
    }
  }

  #pragma omp parallel for collapse(3)
  for (int it = 9; it < n3-11; it++) {
    for (int ix = FAT; ix < n2-FAT; ix++) {
      for (int iz = FAT; iz < n1-FAT; iz++) {
        (*m)[it+1][ix][iz] += //second time deriv
                         //C0t_10*((*fm)[ix][iz]*(*d)[it][ix ][iz])*(*s)[ix][iz] -
       			 (C0t_10*(*fm)[ix][iz]* (*d)[it][ix ][iz ]+ \
                         C1t_10*((*d)[it-1][ix ][iz ] + (*fm)[ix][iz]*(*d)[it + 1][ix ][iz]) + \
                         C2t_10*((*d)[it-2][ix ][iz ] + (*fm)[ix][iz]*(*d)[it + 2][ix ][iz]) + \
                         C3t_10*((*d)[it-3][ix ][iz ] + (*fm)[ix][iz]*(*d)[it + 3][ix ][iz]) + \
                         C4t_10*((*d)[it-4][ix ][iz ] + (*fm)[ix][iz]*(*d)[it + 4][ix ][iz]) + \
                         C5t_10*((*d)[it-5][ix ][iz ] + (*fm)[ix][iz]*(*d)[it + 5][ix ][iz]))*(*s)[ix][iz]  - \
                         //laplacian
                         (C0x *(*d)[it][ix ][iz ] + \
                         C1x * ((*d)[it][ix + 1 ][iz ] + (*d)[it][ix - 1 ][iz ]) + \
                         C2x * ((*d)[it][ix + 2 ][iz ] + (*d)[it][ix - 2 ][iz ]) + \
                         C3x * ((*d)[it][ix + 3 ][iz ] + (*d)[it][ix - 3 ][iz ]) + \
                         C4x * ((*d)[it][ix + 4 ][iz ] + (*d)[it][ix - 4 ][iz ]) + \
                         C5x * ((*d)[it][ix + 5 ][iz ] + (*d)[it][ix - 5 ][iz ]) + \
                         C0z * ((*d)[it][ix ][iz ]) + \
                         C1z * ((*d)[it][ix ][iz + 1 ] + (*d)[it][ix ][iz - 1 ]) + \
                         C2z * ((*d)[it][ix ][iz + 2 ] + (*d)[it][ix ][iz - 2 ]) + \
                         C3z * ((*d)[it][ix ][iz + 3 ] + (*d)[it][ix ][iz - 3 ]) + \
                         C4z * ((*d)[it][ix ][iz + 4 ] + (*d)[it][ix ][iz - 4 ]) + \
                         C5z * ((*d)[it][ix ][iz + 5 ] + (*d)[it][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[it-1][ix ][iz ]-(*d)[it][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[it][ix ][iz ];
      }
    }
  }
}
/*###########################################################################
                          Multi experiment
/*###########################################################################*/
WaveRecon_multi_exp::WaveRecon_multi_exp(const std::shared_ptr<SEP::float4DReg>model,
                         const std::shared_ptr<SEP::float4DReg>data,
                         const std::shared_ptr<SEP::float2DReg>slsqModel,
                         float          U_0,
                         float 					alpha,
			                   int					  spongeWidth) {

  // ensure model and data dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(4).n == data->getHyper()->getAxis(4).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  // ensure velModel matches fmatial dimensions of model and data
  assert(model->getHyper()->getAxis(1).n ==
         slsqModel->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n ==
         slsqModel->getHyper()->getAxis(2).n);

  _spongeWidth=spongeWidth;

  n1 =model->getHyper()->getAxis(1).n; //z
   n2 =model->getHyper()->getAxis(2).n; //x
   n3 =model->getHyper()->getAxis(3).n; //t
   n4 =model->getHyper()->getAxis(4).n; //experiment

  // set domain and range
  setDomainRange(model, data);

  // set up  lapl operator
  // get fmatial sampling
  _db = model->getHyper()->getAxis(1).d;
  _da = model->getHyper()->getAxis(2).d;
  _dt = model->getHyper()->getAxis(3).d;

  // calculate lapl coefficients
  C0z = -2.927222222 / (_db * _db);
  C1z =  1.666666667 / (_db * _db);
  C2z = -0.238095238 / (_db * _db);
  C3z =  0.039682539 / (_db * _db);
  C4z = -0.004960317 / (_db * _db);
  C5z =  0.000317460 / (_db * _db);

  C0x = -2.927222222 / (_da * _da);
  C1x =  1.666666667 / (_da * _da);
  C2x = -0.238095238 / (_da * _da);
  C3x =  0.039682539 / (_da * _da);
  C4x = -0.004960317 / (_da * _da);
  C5x =  0.000317460 / (_da * _da);

  C0t_10 = -2.927222222 / (_dt * _dt);
  C1t_10 =  1.666666667 / (_dt * _dt);
  C2t_10 = -0.23809524 / (_dt * _dt);
  C3t_10 =  0.03968254 / (_dt * _dt);
  C4t_10 = -0.00496032 / (_dt * _dt);
  C5t_10 =  0.00031746 / (_dt * _dt);

  C0t_8 = -2.84722222222 / (_dt * _dt);
  C1t_8 =  1.600000000 / (_dt * _dt);
  C2t_8 = -0.200000000/ (_dt * _dt);
  C3t_8 =  0.02539682539 / (_dt * _dt);
  C4t_8 = -0.00178571428 / (_dt * _dt);

  C0t_6 = -2.72222222222 / (_dt * _dt);
  C1t_6 =  1.500000000 / (_dt * _dt);
  C2t_6 = -0.150000000/ (_dt * _dt);
  C3t_6 =  0.01111111111 / (_dt * _dt);

  C0t_4 = -2.50000000000 / (_dt * _dt);
  C1t_4 =  1.33333333333 / (_dt * _dt);
  C2t_4 = -0.08333333333 / (_dt * _dt);

  C0t_2 = -2.00000000000 / (_dt * _dt);
  C1t_2 =  1.000000000 / (_dt * _dt);

  setDomainRange(model, data);

    _fatMask.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
                                      data->getHyper()->getAxis(2).n));
    _fatMask->set(0);
    for (int ix = FAT; ix < n2-FAT; ix++){
      for (int iz = FAT; iz < n1-FAT; iz++) {
	       (*_fatMask->_mat)[ix][iz] = 1;
      }
    }

    //calculate gamma
    _gamma.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
                                      data->getHyper()->getAxis(2).n));
    _gammaSq.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
                                      data->getHyper()->getAxis(2).n));
    _gamma->set(0.0);
    _gammaSq->set(0.0);
    _U_0 = U_0;
    _alpha = alpha;
    for (int ix = FAT; ix < n2-FAT; ix++){
      for (int iz = FAT; iz < n1-FAT; iz++) {
	      int distToEdge = std::min(std::min(ix-FAT,iz-FAT),std::min(n1-FAT-iz-1,n2-FAT-ix-1));
	      if(distToEdge < _spongeWidth){
          float gamma = _U_0/(std::cosh(_alpha*distToEdge)*std::cosh(_alpha*distToEdge));
	        (*_gamma->_mat)[ix][iz] = 2*gamma/_dt;
	        (*_gammaSq->_mat)[ix][iz] = gamma*gamma;
        }
      }
    }
  // set slowness
  _slsq=slsqModel;
}

void WaveRecon_multi_exp::set_slsq(std::shared_ptr<SEP::float2DReg>slsq){
	_slsq=slsq;
}

void WaveRecon_multi_exp::forward(const bool                         add,
                          const std::shared_ptr<SEP::float4DReg>model,
                          std::shared_ptr<SEP::float4DReg>      data) const {

  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);


  const std::shared_ptr<float4D> m = ((std::dynamic_pointer_cast<float4DReg>(model))->_mat);
  std::shared_ptr<float4D> d = ((std::dynamic_pointer_cast<float4DReg>(data))->_mat);
  std::shared_ptr<float2D> fm = _fatMask->_mat;
  std::shared_ptr<float2D> s = ((std::dynamic_pointer_cast<float2DReg>(_slsq))->_mat);
  std::shared_ptr<float2D> g = _gamma->_mat;
  std::shared_ptr<float2D> gs = _gammaSq->_mat;

  //boundary condition
  #pragma omp parallel for collapse(3)
  for(int is = 0; is < n4; is++) {
    for (int ix = FAT; ix < n2-FAT; ix++) { //x
      for (int iz = FAT; iz < n1-FAT; iz++) { //z
        (*d)[is][0][ix][iz] += //second time deriv
          (C0t_2*(*fm)[ix][iz]* (*m)[is][1][ix ][iz ] + \
            C1t_2*((*fm)[ix][iz] * (*m)[is][1-1][ix ][iz ]+(*m)[is][1+1][ix ][iz]))*(*s)[ix][iz]  - \
            //laplacian
            (C0x * (*fm)[ix][iz]*(*m)[is][1][ix ][iz ] + \
            C1x * ((*fm)[ix+1][iz]*(*m)[is][1][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[is][1][ix - 1 ][iz ]) + \
            C2x * ((*fm)[ix+2][iz]*(*m)[is][1][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[is][1][ix - 2 ][iz ]) + \
            C3x * ((*fm)[ix+3][iz]*(*m)[is][1][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[is][1][ix - 3 ][iz ]) + \
            C4x * ((*fm)[ix+4][iz]*(*m)[is][1][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[is][1][ix - 4 ][iz ]) + \
            C5x * ((*fm)[ix+5][iz]*(*m)[is][1][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[is][1][ix - 5 ][iz ]) + \
            C0z * ((*fm)[ix][iz]*(*m)[is][1][ix ][iz ]) + \
            C1z * ((*fm)[ix][iz+1]*(*m)[is][1][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[is][1][ix ][iz - 1 ]) + \
            C2z * ((*fm)[ix][iz+2]*(*m)[is][1][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[is][1][ix ][iz - 2 ]) + \
            C3z * ((*fm)[ix][iz+3]*(*m)[is][1][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[is][1][ix ][iz - 3 ]) + \
            C4z * ((*fm)[ix][iz+4]*(*m)[is][1][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[is][1][ix ][iz - 4 ]) + \
            C5z * ((*fm)[ix][iz+5]*(*m)[is][1][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[is][1][ix ][iz - 5 ])) + \
      		//sponge first term
      		(*g)[ix][iz]*((*m)[is][2][ix ][iz ]-(*m)[is][1][ix ][iz ]) + \
      		//sponge second term
      		(*gs)[ix][iz]*(*m)[is][1][ix ][iz ];

        (*d)[is][1][ix][iz] += //second time deriv
       		(C0t_4*(*fm)[ix][iz]* (*m)[is][2][ix ][iz ] + \
            C1t_4*((*fm)[ix][iz] * (*m)[is][2-1][ix ][iz ]+(*m)[is][2 + 1][ix ][iz]) + \
            C2t_4*((*fm)[ix][iz] * (*m)[is][2-2][ix ][iz ]+(*m)[is][2 + 2][ix ][iz]))*(*s)[ix][iz]  - \
            //laplacian
            (C0x * (*fm)[ix][iz]*(*m)[is][2][ix ][iz ] + \
            C1x * ((*fm)[ix+1][iz]*(*m)[is][2][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[is][2][ix - 1 ][iz ]) + \
            C2x * ((*fm)[ix+2][iz]*(*m)[is][2][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[is][2][ix - 2 ][iz ]) + \
            C3x * ((*fm)[ix+3][iz]*(*m)[is][2][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[is][2][ix - 3 ][iz ]) + \
            C4x * ((*fm)[ix+4][iz]*(*m)[is][2][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[is][2][ix - 4 ][iz ]) + \
            C5x * ((*fm)[ix+5][iz]*(*m)[is][2][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[is][2][ix - 5 ][iz ]) + \
            C0z * ((*fm)[ix][iz]*(*m)[is][2][ix ][iz ]) + \
            C1z * ((*fm)[ix][iz+1]*(*m)[is][2][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[is][2][ix ][iz - 1 ]) + \
            C2z * ((*fm)[ix][iz+2]*(*m)[is][2][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[is][2][ix ][iz - 2 ]) + \
            C3z * ((*fm)[ix][iz+3]*(*m)[is][2][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[is][2][ix ][iz - 3 ]) + \
            C4z * ((*fm)[ix][iz+4]*(*m)[is][2][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[is][2][ix ][iz - 4 ]) + \
            C5z * ((*fm)[ix][iz+5]*(*m)[is][2][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[is][2][ix ][iz - 5 ])) + \
      		//sponge first term
      		(*g)[ix][iz]*((*m)[is][3][ix ][iz ]-(*m)[is][2][ix ][iz ]) + \
      		//sponge second term
      		(*gs)[ix][iz]*(*m)[is][2][ix ][iz ];

        (*d)[is][2][ix][iz] += //second time deriv
       		(C0t_6*(*fm)[ix][iz]* (*m)[is][3][ix ][iz ] + \
            C1t_6*((*fm)[ix][iz] * (*m)[is][3-1][ix ][iz ]+(*m)[is][3 + 1][ix ][iz]) + \
            C2t_6*((*fm)[ix][iz] * (*m)[is][3-2][ix ][iz ]+(*m)[is][3 + 2][ix ][iz]) + \
            C3t_6*((*fm)[ix][iz] * (*m)[is][3-3][ix ][iz ]+(*m)[is][3 + 3][ix ][iz]))*(*s)[ix][iz]  - \
            //laplacian
            (C0x * (*fm)[ix][iz]*(*m)[is][3][ix ][iz ] + \
            C1x * ((*fm)[ix+1][iz]*(*m)[is][3][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[is][3][ix - 1 ][iz ]) + \
            C2x * ((*fm)[ix+2][iz]*(*m)[is][3][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[is][3][ix - 2 ][iz ]) + \
            C3x * ((*fm)[ix+3][iz]*(*m)[is][3][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[is][3][ix - 3 ][iz ]) + \
            C4x * ((*fm)[ix+4][iz]*(*m)[is][3][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[is][3][ix - 4 ][iz ]) + \
            C5x * ((*fm)[ix+5][iz]*(*m)[is][3][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[is][3][ix - 5 ][iz ]) + \
            C0z * ((*fm)[ix][iz]*(*m)[is][3][ix ][iz ]) + \
            C1z * ((*fm)[ix][iz+1]*(*m)[is][3][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[is][3][ix ][iz - 1 ]) + \
            C2z * ((*fm)[ix][iz+2]*(*m)[is][3][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[is][3][ix ][iz - 2 ]) + \
            C3z * ((*fm)[ix][iz+3]*(*m)[is][3][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[is][3][ix ][iz - 3 ]) + \
            C4z * ((*fm)[ix][iz+4]*(*m)[is][3][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[is][3][ix ][iz - 4 ]) + \
            C5z * ((*fm)[ix][iz+5]*(*m)[is][3][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[is][3][ix ][iz - 5 ])) + \
      		//sponge first term
      		(*g)[ix][iz]*((*m)[is][4][ix ][iz ]-(*m)[is][3][ix ][iz ]) + \
      		//sponge second term
      		(*gs)[ix][iz]*(*m)[is][3][ix ][iz ];

        (*d)[is][3][ix][iz] += //second time deriv
       		(C0t_8*(*fm)[ix][iz]* (*m)[is][4][ix ][iz ] + \
            C1t_8*((*fm)[ix][iz] * (*m)[is][4-1][ix ][iz ]+(*m)[is][4 + 1][ix ][iz]) + \
            C2t_8*((*fm)[ix][iz] * (*m)[is][4-2][ix ][iz ]+(*m)[is][4 + 2][ix ][iz]) + \
            C3t_8*((*fm)[ix][iz] * (*m)[is][4-3][ix ][iz ]+(*m)[is][4 + 3][ix ][iz])  + \
            C4t_8*((*fm)[ix][iz] * (*m)[is][4-4][ix ][iz ]+(*m)[is][4 + 4][ix ][iz]))*(*s)[ix][iz]  - \
            //laplacian
            (C0x * (*fm)[ix][iz]*(*m)[is][4][ix ][iz ] + \
            C1x * ((*fm)[ix+1][iz]*(*m)[is][4][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[is][4][ix - 1 ][iz ]) + \
            C2x * ((*fm)[ix+2][iz]*(*m)[is][4][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[is][4][ix - 2 ][iz ]) + \
            C3x * ((*fm)[ix+3][iz]*(*m)[is][4][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[is][4][ix - 3 ][iz ]) + \
            C4x * ((*fm)[ix+4][iz]*(*m)[is][4][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[is][4][ix - 4 ][iz ]) + \
            C5x * ((*fm)[ix+5][iz]*(*m)[is][4][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[is][4][ix - 5 ][iz ]) + \
            C0z * ((*fm)[ix][iz]*(*m)[is][4][ix ][iz ]) + \
            C1z * ((*fm)[ix][iz+1]*(*m)[is][4][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[is][4][ix ][iz - 1 ]) + \
            C2z * ((*fm)[ix][iz+2]*(*m)[is][4][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[is][4][ix ][iz - 2 ]) + \
            C3z * ((*fm)[ix][iz+3]*(*m)[is][4][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[is][4][ix ][iz - 3 ]) + \
            C4z * ((*fm)[ix][iz+4]*(*m)[is][4][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[is][4][ix ][iz - 4 ]) + \
            C5z * ((*fm)[ix][iz+5]*(*m)[is][4][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[is][4][ix ][iz - 5 ])) + \
      		//sponge first term
      		(*g)[ix][iz]*((*m)[is][5][ix ][iz ]-(*m)[is][4][ix ][iz ]) + \
      		//sponge second term
      		(*gs)[ix][iz]*(*m)[is][4][ix ][iz ];

        (*d)[is][n3-6][ix][iz] += //second time deriv
       		(C0t_8*(*fm)[ix][iz]* (*m)[is][n3-5][ix ][iz ]+ \
            C1t_8*((*fm)[ix][iz] * (*m)[is][n3-5-1][ix ][iz ]+(*m)[is][n3-5 + 1][ix ][iz]) + \
            C2t_8*((*fm)[ix][iz] * (*m)[is][n3-5-2][ix ][iz ]+(*m)[is][n3-5 + 2][ix ][iz]) + \
            C3t_8*((*fm)[ix][iz] * (*m)[is][n3-5-3][ix ][iz ]+(*m)[is][n3-5 + 3][ix ][iz])  + \
            C4t_8*((*fm)[ix][iz] * (*m)[is][n3-5-4][ix ][iz ]+(*m)[is][n3-5 + 4][ix ][iz]))*(*s)[ix][iz]  - \
            //laplacian
            (C0x * (*fm)[ix][iz]*(*m)[is][n3-5][ix ][iz ] + \
            C1x * ((*fm)[ix+1][iz]*(*m)[is][n3-5][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[is][n3-5][ix - 1 ][iz ]) + \
            C2x * ((*fm)[ix+2][iz]*(*m)[is][n3-5][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[is][n3-5][ix - 2 ][iz ]) + \
            C3x * ((*fm)[ix+3][iz]*(*m)[is][n3-5][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[is][n3-5][ix - 3 ][iz ]) + \
            C4x * ((*fm)[ix+4][iz]*(*m)[is][n3-5][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[is][n3-5][ix - 4 ][iz ]) + \
            C5x * ((*fm)[ix+5][iz]*(*m)[is][n3-5][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[is][n3-5][ix - 5 ][iz ]) + \
            C0z * ((*fm)[ix][iz]*(*m)[is][n3-5][ix ][iz ]) + \
            C1z * ((*fm)[ix][iz+1]*(*m)[is][n3-5][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[is][n3-5][ix ][iz - 1 ]) + \
            C2z * ((*fm)[ix][iz+2]*(*m)[is][n3-5][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[is][n3-5][ix ][iz - 2 ]) + \
            C3z * ((*fm)[ix][iz+3]*(*m)[is][n3-5][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[is][n3-5][ix ][iz - 3 ]) + \
            C4z * ((*fm)[ix][iz+4]*(*m)[is][n3-5][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[is][n3-5][ix ][iz - 4 ]) + \
            C5z * ((*fm)[ix][iz+5]*(*m)[is][n3-5][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[is][n3-5][ix ][iz - 5 ])) + \
      		//sponge first term
      		(*g)[ix][iz]*((*m)[is][n3-4][ix ][iz ]-(*m)[is][n3-5][ix ][iz ]) + \
      		//sponge second term
      		(*gs)[ix][iz]*(*m)[is][n3-5][ix ][iz ];

        (*d)[is][n3-5][ix][iz] += //second time deriv
       		(C0t_6*(*fm)[ix][iz]* (*m)[is][n3-4][ix ][iz ] + \
            C1t_6*((*fm)[ix][iz] * (*m)[is][n3-4-1][ix ][iz ]+(*m)[is][n3-4 + 1][ix ][iz]) + \
            C2t_6*((*fm)[ix][iz] * (*m)[is][n3-4-2][ix ][iz ]+(*m)[is][n3-4 + 2][ix ][iz]) + \
            C3t_6*((*fm)[ix][iz] * (*m)[is][n3-4-3][ix ][iz ]+(*m)[is][n3-4 + 3][ix ][iz]))*(*s)[ix][iz]  - \
            //laplacian
            (C0x * (*fm)[ix][iz]*(*m)[is][n3-4][ix ][iz ] + \
            C1x * ((*fm)[ix+1][iz]*(*m)[is][n3-4][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[is][n3-4][ix - 1 ][iz ]) + \
            C2x * ((*fm)[ix+2][iz]*(*m)[is][n3-4][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[is][n3-4][ix - 2 ][iz ]) + \
            C3x * ((*fm)[ix+3][iz]*(*m)[is][n3-4][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[is][n3-4][ix - 3 ][iz ]) + \
            C4x * ((*fm)[ix+4][iz]*(*m)[is][n3-4][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[is][n3-4][ix - 4 ][iz ]) + \
            C5x * ((*fm)[ix+5][iz]*(*m)[is][n3-4][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[is][n3-4][ix - 5 ][iz ]) + \
            C0z * ((*fm)[ix][iz]*(*m)[is][n3-4][ix ][iz ]) + \
            C1z * ((*fm)[ix][iz+1]*(*m)[is][n3-4][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[is][n3-4][ix ][iz - 1 ]) + \
            C2z * ((*fm)[ix][iz+2]*(*m)[is][n3-4][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[is][n3-4][ix ][iz - 2 ]) + \
            C3z * ((*fm)[ix][iz+3]*(*m)[is][n3-4][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[is][n3-4][ix ][iz - 3 ]) + \
            C4z * ((*fm)[ix][iz+4]*(*m)[is][n3-4][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[is][n3-4][ix ][iz - 4 ]) + \
            C5z * ((*fm)[ix][iz+5]*(*m)[is][n3-4][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[is][n3-4][ix ][iz - 5 ])) + \
      		//sponge first term
      		(*g)[ix][iz]*((*m)[is][n3-3][ix ][iz ]-(*m)[is][n3-4][ix ][iz ]) + \
      		//sponge second term
      		(*gs)[ix][iz]*(*m)[is][n3-4][ix ][iz ];

        (*d)[is][n3-4][ix][iz] += //second time deriv
       		(C0t_4*(*fm)[ix][iz]* (*m)[is][n3-3][ix ][iz ] + \
            C1t_4*((*fm)[ix][iz] * (*m)[is][n3-3-1][ix ][iz ]+(*m)[is][n3-3 + 1][ix ][iz]) + \
            C2t_4*((*fm)[ix][iz] * (*m)[is][n3-3-2][ix ][iz ]+(*m)[is][n3-3 + 2][ix ][iz]))*(*s)[ix][iz]  - \
            //laplacian
            (C0x * (*fm)[ix][iz]*(*m)[is][n3-3][ix ][iz ] + \
            C1x * ((*fm)[ix+1][iz]*(*m)[is][n3-3][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[is][n3-3][ix - 1 ][iz ]) + \
            C2x * ((*fm)[ix+2][iz]*(*m)[is][n3-3][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[is][n3-3][ix - 2 ][iz ]) + \
            C3x * ((*fm)[ix+3][iz]*(*m)[is][n3-3][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[is][n3-3][ix - 3 ][iz ]) + \
            C4x * ((*fm)[ix+4][iz]*(*m)[is][n3-3][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[is][n3-3][ix - 4 ][iz ]) + \
            C5x * ((*fm)[ix+5][iz]*(*m)[is][n3-3][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[is][n3-3][ix - 5 ][iz ]) + \
            C0z * ((*fm)[ix][iz]*(*m)[is][n3-3][ix ][iz ]) + \
            C1z * ((*fm)[ix][iz+1]*(*m)[is][n3-3][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[is][n3-3][ix ][iz - 1 ]) + \
            C2z * ((*fm)[ix][iz+2]*(*m)[is][n3-3][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[is][n3-3][ix ][iz - 2 ]) + \
            C3z * ((*fm)[ix][iz+3]*(*m)[is][n3-3][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[is][n3-3][ix ][iz - 3 ]) + \
            C4z * ((*fm)[ix][iz+4]*(*m)[is][n3-3][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[is][n3-3][ix ][iz - 4 ]) + \
            C5z * ((*fm)[ix][iz+5]*(*m)[is][n3-3][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[is][n3-3][ix ][iz - 5 ])) + \
      		//sponge first term
      		(*g)[ix][iz]*((*m)[is][n3-2][ix ][iz ]-(*m)[is][n3-3][ix ][iz ]) + \
      		//sponge second term
      		(*gs)[ix][iz]*(*m)[is][n3-3][ix ][iz ];

        (*d)[is][n3-3][ix][iz] += //second time deriv
       		(C0t_2*(*fm)[ix][iz]* (*m)[is][n3-2][ix ][iz ] + \
            C1t_2*((*fm)[ix][iz] * (*m)[is][n3-2-1][ix ][iz ]+(*m)[is][n3-2 + 1][ix ][iz]))*(*s)[ix][iz]  - \
            //laplacian
            (C0x * (*fm)[ix][iz]*(*m)[is][n3-2][ix ][iz ] + \
            C1x * ((*fm)[ix+1][iz]*(*m)[is][n3-2][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[is][n3-2][ix - 1 ][iz ]) + \
            C2x * ((*fm)[ix+2][iz]*(*m)[is][n3-2][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[is][n3-2][ix - 2 ][iz ]) + \
            C3x * ((*fm)[ix+3][iz]*(*m)[is][n3-2][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[is][n3-2][ix - 3 ][iz ]) + \
            C4x * ((*fm)[ix+4][iz]*(*m)[is][n3-2][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[is][n3-2][ix - 4 ][iz ]) + \
            C5x * ((*fm)[ix+5][iz]*(*m)[is][n3-2][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[is][n3-2][ix - 5 ][iz ]) + \
            C0z * ((*fm)[ix][iz]*(*m)[is][n3-2][ix ][iz ]) + \
            C1z * ((*fm)[ix][iz+1]*(*m)[is][n3-2][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[is][n3-2][ix ][iz - 1 ]) + \
            C2z * ((*fm)[ix][iz+2]*(*m)[is][n3-2][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[is][n3-2][ix ][iz - 2 ]) + \
            C3z * ((*fm)[ix][iz+3]*(*m)[is][n3-2][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[is][n3-2][ix ][iz - 3 ]) + \
            C4z * ((*fm)[ix][iz+4]*(*m)[is][n3-2][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[is][n3-2][ix ][iz - 4 ]) + \
            C5z * ((*fm)[ix][iz+5]*(*m)[is][n3-2][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[is][n3-2][ix ][iz - 5 ])) + \
      		//sponge first term
      		(*g)[ix][iz]*((*m)[is][n3-1][ix ][iz ]-(*m)[is][n3-2][ix ][iz ]) + \
      		//sponge second term
      		(*gs)[ix][iz]*(*m)[is][n3-2][ix ][iz ];
            (*d)[is][n3-2][ix][iz] += (*m)[is][0 ][ix ][iz]-(*m)[is][1 ][ix ][iz];
            (*d)[is][n3-1][ix][iz] += (*m)[is][1 ][ix ][iz];
      }
    }
  }
  #pragma omp parallel for collapse(4)
  for(int is = 0; is < n4; is++) { // experiment
    for (int it = 5; it < n3-5; it++) { //time
      for (int ix = FAT; ix < n2-FAT; ix++) { //x
        for (int iz = FAT; iz < n1-FAT; iz++) { //z
          (*d)[is][it-1][ix][iz] +=//second time deriv
  		//C0t_10*((*fm)[ix][iz] * (*m)[is][it][ix ][iz ])*(*s)[ix][iz]  -
         		(C0t_10*(*fm)[ix][iz]* (*m)[is][it][ix ][iz ] + \
              C1t_10*((*fm)[ix][iz] * (*m)[is][it-1][ix ][iz ]+(*m)[is][it + 1][ix ][iz]) + \
              C2t_10*((*fm)[ix][iz] * (*m)[is][it-2][ix ][iz ]+(*m)[is][it + 2][ix ][iz]) + \
              C3t_10*((*fm)[ix][iz] * (*m)[is][it-3][ix ][iz ]+(*m)[is][it + 3][ix ][iz]) + \
              C4t_10*((*fm)[ix][iz] * (*m)[is][it-4][ix ][iz ]+(*m)[is][it + 4][ix ][iz]) + \
              C5t_10*((*fm)[ix][iz] * (*m)[is][it-5][ix ][iz ]+(*m)[is][it + 5][ix ][iz]))*(*s)[ix][iz]  - \
              //laplacian
              (C0x * (*fm)[ix][iz]*(*m)[is][it][ix ][iz ] + \
              C1x * ((*fm)[ix+1][iz]*(*m)[is][it][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*m)[is][it][ix - 1 ][iz ]) + \
              C2x * ((*fm)[ix+2][iz]*(*m)[is][it][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*m)[is][it][ix - 2 ][iz ]) + \
              C3x * ((*fm)[ix+3][iz]*(*m)[is][it][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*m)[is][it][ix - 3 ][iz ]) + \
              C4x * ((*fm)[ix+4][iz]*(*m)[is][it][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*m)[is][it][ix - 4 ][iz ]) + \
              C5x * ((*fm)[ix+5][iz]*(*m)[is][it][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*m)[is][it][ix - 5 ][iz ]) + \
              C0z * ((*fm)[ix][iz]*(*m)[is][it][ix ][iz ]) + \
              C1z * ((*fm)[ix][iz+1]*(*m)[is][it][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*m)[is][it][ix ][iz - 1 ]) + \
              C2z * ((*fm)[ix][iz+2]*(*m)[is][it][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*m)[is][it][ix ][iz - 2 ]) + \
              C3z * ((*fm)[ix][iz+3]*(*m)[is][it][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*m)[is][it][ix ][iz - 3 ]) + \
              C4z * ((*fm)[ix][iz+4]*(*m)[is][it][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*m)[is][it][ix ][iz - 4 ]) + \
              C5z * ((*fm)[ix][iz+5]*(*m)[is][it][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*m)[is][it][ix ][iz - 5 ])) + \
        		//sponge first term
        		(*g)[ix][iz]*((*m)[is][it+1][ix ][iz ]-(*m)[is][it][ix ][iz ]) + \
        		//sponge second term
        		(*gs)[ix][iz]*(*m)[is][it][ix ][iz ];
        }
      }
    }
  }

}

void WaveRecon_multi_exp::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float4DReg>      model,
                          const std::shared_ptr<SEP::float4DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float4D> m = ((std::dynamic_pointer_cast<float4DReg>(model))->_mat);
  const std::shared_ptr<float4D> d = ((std::dynamic_pointer_cast<float4DReg>(data))->_mat);
  std::shared_ptr<float2D> s = ((std::dynamic_pointer_cast<float2DReg>(_slsq))->_mat);
  std::shared_ptr<float2D> fm = _fatMask->_mat;
  std::shared_ptr<float2D> g = _gamma->_mat;
  std::shared_ptr<float2D> gs = _gammaSq->_mat;


  #pragma omp parallel for collapse(3)
  for(int is = 0; is < n4; is++) {
    for (int ix = FAT; ix < n2-FAT; ix++) { //x
      for (int iz = FAT; iz < n1-FAT; iz++) { //z
        (*m)[is][0][ix][iz] +=   //second time deriv
          (C1t_2*((*fm)[ix][iz]*(*d)[is][0][ix ][iz]) + \
          C2t_4*((*fm)[ix][iz]*(*d)[is][1][ix ][iz]) + \
          C3t_6*((*fm)[ix][iz]*(*d)[is][2][ix ][iz]) + \
          C4t_8*((*fm)[ix][iz]*(*d)[is][3][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][4][ix ][iz]))*(*s)[ix][iz] + \
          //initial condition
          (*d)[is][n3-2][ix ][iz ];

        (*m)[is][1][ix][iz] +=   //second time deriv
          (C0t_2*((*fm)[ix][iz]*(*d)[is][0][ix ][iz]) + \
          C1t_4*((*fm)[ix][iz]*(*d)[is][1][ix ][iz]) + \
          C2t_6*((*fm)[ix][iz]*(*d)[is][2][ix ][iz]) + \
          C3t_8*((*fm)[ix][iz]*(*d)[is][3][ix ][iz]) + \
          C4t_10*((*fm)[ix][iz]*(*d)[is][4][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][0][ix ][iz ] + \
          C1x * ((*d)[is][0][ix + 1 ][iz ] + (*d)[is][0][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][0][ix + 2 ][iz ] + (*d)[is][0][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][0][ix + 3 ][iz ] + (*d)[is][0][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][0][ix + 4 ][iz ] + (*d)[is][0][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][0][ix + 5 ][iz ] + (*d)[is][0][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][0][ix ][iz ]) + \
          C1z * ((*d)[is][0][ix ][iz + 1 ] + (*d)[is][0][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][0][ix ][iz + 2 ] + (*d)[is][0][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][0][ix ][iz + 3 ] + (*d)[is][0][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][0][ix ][iz + 4 ] + (*d)[is][0][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][0][ix ][iz + 5 ] + (*d)[is][0][ix ][iz - 5 ]))*(*fm)[ix][iz] +
          //initial condition
          (*d)[is][n3-1][ix ][iz ] - (*d)[is][n3-2][ix ][iz ]+ \
          //sponge first term
          (*g)[ix][iz]*(-1)*(*d)[is][0][ix ][iz ] + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][0][ix ][iz ];

        (*m)[is][2][ix][iz] +=   //second time deriv
          (C0t_4*((*fm)[ix][iz]*(*d)[is][1][ix ][iz]) + \
          C1t_2*(*d)[is][1-1][ix ][iz] + C1t_6*((*fm)[ix][iz]*(*d)[is][1+1][ix ][iz]) + \
          C2t_8*((*fm)[ix][iz]*(*d)[is][1+2][ix ][iz]) + \
          C3t_10*((*fm)[ix][iz]*(*d)[is][1+3][ix ][iz]) + \
          C4t_10*((*fm)[ix][iz]*(*d)[is][1+4][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][1+5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][1][ix ][iz ] + \
          C1x * ((*d)[is][1][ix + 1 ][iz ] + (*d)[is][1][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][1][ix + 2 ][iz ] + (*d)[is][1][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][1][ix + 3 ][iz ] + (*d)[is][1][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][1][ix + 4 ][iz ] + (*d)[is][1][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][1][ix + 5 ][iz ] + (*d)[is][1][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][1][ix ][iz ]) + \
          C1z * ((*d)[is][1][ix ][iz + 1 ] + (*d)[is][1][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][1][ix ][iz + 2 ] + (*d)[is][1][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][1][ix ][iz + 3 ] + (*d)[is][1][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][1][ix ][iz + 4 ] + (*d)[is][1][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][1][ix ][iz + 5 ] + (*d)[is][1][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][0][ix ][iz ]-(*d)[is][1][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][1][ix ][iz ];

        (*m)[is][3][ix][iz] +=   //second time deriv
          (C0t_6*((*fm)[ix][iz]*(*d)[is][2][ix ][iz]) + \
          C1t_4*(*d)[is][1][ix ][iz] + C1t_8*((*fm)[ix][iz]*(*d)[is][3][ix ][iz]) + \
          C2t_10*((*fm)[ix][iz]*(*d)[is][4][ix ][iz]) + \
          C3t_10*((*fm)[ix][iz]*(*d)[is][5][ix ][iz]) + \
          C4t_10*((*fm)[ix][iz]*(*d)[is][6][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][7][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][2][ix ][iz ] + \
          C1x * ((*d)[is][2][ix + 1 ][iz ] + (*d)[is][2][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][2][ix + 2 ][iz ] + (*d)[is][2][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][2][ix + 3 ][iz ] + (*d)[is][2][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][2][ix + 4 ][iz ] + (*d)[is][2][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][2][ix + 5 ][iz ] + (*d)[is][2][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][2][ix ][iz ]) + \
          C1z * ((*d)[is][2][ix ][iz + 1 ] + (*d)[is][2][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][2][ix ][iz + 2 ] + (*d)[is][2][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][2][ix ][iz + 3 ] + (*d)[is][2][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][2][ix ][iz + 4 ] + (*d)[is][2][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][2][ix ][iz + 5 ] + (*d)[is][2][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][1][ix ][iz ]-(*d)[is][2][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][2][ix ][iz ];

        (*m)[is][4][ix][iz] +=   //second time deriv
          (C0t_8*((*fm)[ix][iz]*(*d)[is][3][ix ][iz]) + \
          C1t_6*(*d)[is][2][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][4][ix ][iz]) + \
          C2t_4*(*d)[is][1][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[is][5][ix ][iz]) + \
          C3t_10*((*fm)[ix][iz]*(*d)[is][6][ix ][iz]) + \
          C4t_10*((*fm)[ix][iz]*(*d)[is][7][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][8][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][3][ix ][iz ] + \
          C1x * ((*d)[is][3][ix + 1 ][iz ] + (*d)[is][3][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][3][ix + 2 ][iz ] + (*d)[is][3][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][3][ix + 3 ][iz ] + (*d)[is][3][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][3][ix + 4 ][iz ] + (*d)[is][3][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][3][ix + 5 ][iz ] + (*d)[is][3][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][3][ix ][iz ]) + \
          C1z * ((*d)[is][3][ix ][iz + 1 ] + (*d)[is][3][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][3][ix ][iz + 2 ] + (*d)[is][3][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][3][ix ][iz + 3 ] + (*d)[is][3][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][3][ix ][iz + 4 ] + (*d)[is][3][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][3][ix ][iz + 5 ] + (*d)[is][3][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][2][ix ][iz ]-(*d)[is][3][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][3][ix ][iz ];

        (*m)[is][5][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][4][ix ][iz]) + \
          C1t_8*(*d)[is][3][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][5][ix ][iz]) + \
          C2t_6*(*d)[is][2][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[is][6][ix ][iz]) + \
          C3t_10*((*fm)[ix][iz]*(*d)[is][7][ix ][iz]) + \
          C4t_10*((*fm)[ix][iz]*(*d)[is][8][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][9][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][4][ix ][iz ] + \
          C1x * ((*d)[is][4][ix + 1 ][iz ] + (*d)[is][4][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][4][ix + 2 ][iz ] + (*d)[is][4][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][4][ix + 3 ][iz ] + (*d)[is][4][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][4][ix + 4 ][iz ] + (*d)[is][4][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][4][ix + 5 ][iz ] + (*d)[is][4][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][4][ix ][iz ]) + \
          C1z * ((*d)[is][4][ix ][iz + 1 ] + (*d)[is][4][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][4][ix ][iz + 2 ] + (*d)[is][4][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][4][ix ][iz + 3 ] + (*d)[is][4][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][4][ix ][iz + 4 ] + (*d)[is][4][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][4][ix ][iz + 5 ] + (*d)[is][4][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][3][ix ][iz ]-(*d)[is][4][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][4][ix ][iz ];

        (*m)[is][6][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][5][ix ][iz]) + \
          C1t_10*(*d)[is][4][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][6][ix ][iz]) + \
          C2t_8*(*d)[is][3][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[is][7][ix ][iz]) + \
          C3t_6*(*d)[is][2][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[is][8][ix ][iz]) + \
          C4t_10*((*fm)[ix][iz]*(*d)[is][9][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][10][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][5][ix ][iz ] + \
          C1x * ((*d)[is][5][ix + 1 ][iz ] + (*d)[is][5][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][5][ix + 2 ][iz ] + (*d)[is][5][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][5][ix + 3 ][iz ] + (*d)[is][5][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][5][ix + 4 ][iz ] + (*d)[is][5][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][5][ix + 5 ][iz ] + (*d)[is][5][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][5][ix ][iz ]) + \
          C1z * ((*d)[is][5][ix ][iz + 1 ] + (*d)[is][5][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][5][ix ][iz + 2 ] + (*d)[is][5][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][5][ix ][iz + 3 ] + (*d)[is][5][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][5][ix ][iz + 4 ] + (*d)[is][5][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][5][ix ][iz + 5 ] + (*d)[is][5][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][4][ix ][iz ]-(*d)[is][5][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][5][ix ][iz ];

        (*m)[is][7][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][6][ix ][iz])+ \
          C1t_10*(*d)[is][5][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][7][ix ][iz]) + \
          C2t_10*(*d)[is][4][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[is][8][ix ][iz]) + \
          C3t_8*(*d)[is][3][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[is][9][ix ][iz]) + \
          C4t_10*((*fm)[ix][iz]*(*d)[is][10][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][11][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][6][ix ][iz ] + \
          C1x * ((*d)[is][6][ix + 1 ][iz ] + (*d)[is][6][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][6][ix + 2 ][iz ] + (*d)[is][6][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][6][ix + 3 ][iz ] + (*d)[is][6][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][6][ix + 4 ][iz ] + (*d)[is][6][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][6][ix + 5 ][iz ] + (*d)[is][6][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][6][ix ][iz ]) + \
          C1z * ((*d)[is][6][ix ][iz + 1 ] + (*d)[is][6][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][6][ix ][iz + 2 ] + (*d)[is][6][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][6][ix ][iz + 3 ] + (*d)[is][6][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][6][ix ][iz + 4 ] + (*d)[is][6][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][6][ix ][iz + 5 ] + (*d)[is][6][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][5][ix ][iz ]-(*d)[is][6][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][6][ix ][iz ];

        (*m)[is][8][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][7][ix ][iz]) + \
          C1t_10*(*d)[is][6][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][8][ix ][iz]) + \
          C2t_10*(*d)[is][5][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[is][9][ix ][iz]) + \
          C3t_10*(*d)[is][4][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[is][10][ix ][iz]) + \
          C4t_8*(*d)[is][3][ix ][iz] + C4t_10*((*fm)[ix][iz]*(*d)[is][11][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][12][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][7][ix ][iz ]  + \
          C1x * ((*d)[is][7][ix + 1 ][iz ] + (*d)[is][7][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][7][ix + 2 ][iz ] + (*d)[is][7][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][7][ix + 3 ][iz ] + (*d)[is][7][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][7][ix + 4 ][iz ] + (*d)[is][7][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][7][ix + 5 ][iz ] + (*d)[is][7][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][7][ix ][iz ]) + \
          C1z * ((*d)[is][7][ix ][iz + 1 ] + (*d)[is][7][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][7][ix ][iz + 2 ] + (*d)[is][7][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][7][ix ][iz + 3 ] + (*d)[is][7][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][7][ix ][iz + 4 ] + (*d)[is][7][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][7][ix ][iz + 5 ] + (*d)[is][7][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][6][ix ][iz ]-(*d)[is][7][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][7][ix ][iz ];

        (*m)[is][9][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][8][ix ][iz]) + \
          C1t_10*(*d)[is][7][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][9][ix ][iz]) + \
          C2t_10*(*d)[is][6][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[is][10][ix ][iz]) + \
          C3t_10*(*d)[is][5][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[is][11][ix ][iz]) + \
          C4t_10*(*d)[is][4][ix ][iz] + C4t_10*((*fm)[ix][iz]*(*d)[is][12][ix ][iz]) + \
          C5t_10*((*fm)[ix][iz]*(*d)[is][13][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][8][ix ][iz ] + \
          C1x * ((*d)[is][8][ix + 1 ][iz ] + (*d)[is][8][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][8][ix + 2 ][iz ] + (*d)[is][8][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][8][ix + 3 ][iz ] + (*d)[is][8][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][8][ix + 4 ][iz ] + (*d)[is][8][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][8][ix + 5 ][iz ] + (*d)[is][8][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][8][ix ][iz ]) + \
          C1z * ((*d)[is][8][ix ][iz + 1 ] + (*d)[is][8][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][8][ix ][iz + 2 ] + (*d)[is][8][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][8][ix ][iz + 3 ] + (*d)[is][8][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][8][ix ][iz + 4 ] + (*d)[is][8][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][8][ix ][iz + 5 ] + (*d)[is][8][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][7][ix ][iz ]-(*d)[is][8][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][8][ix ][iz ];

        (*m)[is][n3-10][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][n3-11][ix ][iz]) + \
          C1t_10*(*d)[is][n3-11-1][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][n3-11+1][ix ][iz]) + \
          C2t_10*(*d)[is][n3-11-2][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[is][n3-11+2][ix ][iz]) + \
          C3t_10*(*d)[is][n3-11-3][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[is][n3-11+3][ix ][iz]) + \
          C4t_10*(*d)[is][n3-11-4][ix ][iz] + C4t_10*((*fm)[ix][iz]*(*d)[is][n3-11+4][ix ][iz]) + \
          C5t_10*((*d)[is][n3-11-5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][n3-11][ix ][iz ] + \
          C1x * ((*d)[is][n3-11][ix + 1 ][iz ] + (*d)[is][n3-11][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][n3-11][ix + 2 ][iz ] + (*d)[is][n3-11][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][n3-11][ix + 3 ][iz ] + (*d)[is][n3-11][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][n3-11][ix + 4 ][iz ] + (*d)[is][n3-11][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][n3-11][ix + 5 ][iz ] + (*d)[is][n3-11][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][n3-11][ix ][iz ]) + \
          C1z * ((*d)[is][n3-11][ix ][iz + 1 ] + (*d)[is][n3-11][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][n3-11][ix ][iz + 2 ] + (*d)[is][n3-11][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][n3-11][ix ][iz + 3 ] + (*d)[is][n3-11][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][n3-11][ix ][iz + 4 ] + (*d)[is][n3-11][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][n3-11][ix ][iz + 5 ] + (*d)[is][n3-11][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][n3-12][ix ][iz ]-(*d)[is][n3-11][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][n3-11][ix ][iz ];

        (*m)[is][n3-9][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][n3-10][ix ][iz]) + \
          C1t_10*(*d)[is][n3-10-1][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][n3-10+1][ix ][iz]) + \
          C2t_10*(*d)[is][n3-10-2][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[is][n3-10+2][ix ][iz]) + \
          C3t_10*(*d)[is][n3-10-3][ix ][iz] + C3t_10*((*fm)[ix][iz]*(*d)[is][n3-10+3][ix ][iz]) + \
          C4t_10*(*d)[is][n3-10-4][ix ][iz] + C4t_8*((*fm)[ix][iz]*(*d)[is][n3-10+4][ix ][iz]) + \
          C5t_10*((*d)[is][n3-10-5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][n3-10][ix ][iz ] + \
          C1x * ((*d)[is][n3-10][ix + 1 ][iz ] + (*d)[is][n3-10][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][n3-10][ix + 2 ][iz ] + (*d)[is][n3-10][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][n3-10][ix + 3 ][iz ] + (*d)[is][n3-10][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][n3-10][ix + 4 ][iz ] + (*d)[is][n3-10][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][n3-10][ix + 5 ][iz ] + (*d)[is][n3-10][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][n3-10][ix ][iz ]) + \
          C1z * ((*d)[is][n3-10][ix ][iz + 1 ] + (*d)[is][n3-10][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][n3-10][ix ][iz + 2 ] + (*d)[is][n3-10][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][n3-10][ix ][iz + 3 ] + (*d)[is][n3-10][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][n3-10][ix ][iz + 4 ] + (*d)[is][n3-10][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][n3-10][ix ][iz + 5 ] + (*d)[is][n3-10][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][n3-11][ix ][iz ]-(*d)[is][n3-10][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][n3-10][ix ][iz ];

        (*m)[is][n3-8][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][n3-9][ix ][iz]) + \
          C1t_10*(*d)[is][n3-9-1][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][n3-9+1][ix ][iz]) + \
          C2t_10*(*d)[is][n3-9-2][ix ][iz] + C2t_10*((*fm)[ix][iz]*(*d)[is][n3-9+2][ix ][iz]) + \
          C3t_10*(*d)[is][n3-9-3][ix ][iz] + C3t_8*((*fm)[ix][iz]*(*d)[is][n3-9+3][ix ][iz]) + \
          C4t_10*(*d)[is][n3-9-4][ix ][iz] + \
          C5t_10*((*d)[is][n3-9-5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][n3-9][ix ][iz ] + \
          C1x * ((*d)[is][n3-9][ix + 1 ][iz ] + (*d)[is][n3-9][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][n3-9][ix + 2 ][iz ] + (*d)[is][n3-9][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][n3-9][ix + 3 ][iz ] + (*d)[is][n3-9][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][n3-9][ix + 4 ][iz ] + (*d)[is][n3-9][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][n3-9][ix + 5 ][iz ] + (*d)[is][n3-9][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][n3-9][ix ][iz ]) + \
          C1z * ((*d)[is][n3-9][ix ][iz + 1 ] + (*d)[is][n3-9][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][n3-9][ix ][iz + 2 ] + (*d)[is][n3-9][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][n3-9][ix ][iz + 3 ] + (*d)[is][n3-9][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][n3-9][ix ][iz + 4 ] + (*d)[is][n3-9][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][n3-9][ix ][iz + 5 ] + (*d)[is][n3-9][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][n3-10][ix ][iz ]-(*d)[is][n3-9][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][n3-9][ix ][iz ];

        (*m)[is][n3-7][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][n3-8][ix ][iz]) + \
          C1t_10*(*d)[is][n3-8-1][ix ][iz] + C1t_10*((*fm)[ix][iz]*(*d)[is][n3-8+1][ix ][iz]) + \
          C2t_10*(*d)[is][n3-8-2][ix ][iz] + C2t_8*((*fm)[ix][iz]*(*d)[is][n3-8+2][ix ][iz]) + \
          C3t_10*(*d)[is][n3-8-3][ix ][iz] + C3t_6*((*fm)[ix][iz]*(*d)[is][n3-8+3][ix ][iz]) + \
          C4t_10*(*d)[is][n3-8-4][ix ][iz] + \
          C5t_10*((*d)[is][n3-8-5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][n3-8][ix ][iz ]  + \
          C1x * ((*d)[is][n3-8][ix + 1 ][iz ] + (*d)[is][n3-8][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][n3-8][ix + 2 ][iz ] + (*d)[is][n3-8][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][n3-8][ix + 3 ][iz ] + (*d)[is][n3-8][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][n3-8][ix + 4 ][iz ] + (*d)[is][n3-8][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][n3-8][ix + 5 ][iz ] + (*d)[is][n3-8][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][n3-8][ix ][iz ]) + \
          C1z * ((*d)[is][n3-8][ix ][iz + 1 ] + (*d)[is][n3-8][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][n3-8][ix ][iz + 2 ] + (*d)[is][n3-8][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][n3-8][ix ][iz + 3 ] + (*d)[is][n3-8][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][n3-8][ix ][iz + 4 ] + (*d)[is][n3-8][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][n3-8][ix ][iz + 5 ] + (*d)[is][n3-8][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][n3-9][ix ][iz ]-(*d)[is][n3-8][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][n3-8][ix ][iz ];

        (*m)[is][n3-6][ix][iz] +=   //second time deriv
          (C0t_10*((*fm)[ix][iz]*(*d)[is][n3-7][ix ][iz]) + \
          C1t_10*(*d)[is][n3-7-1][ix ][iz] + C1t_8*((*fm)[ix][iz]*(*d)[is][n3-7+1][ix ][iz]) + \
          C2t_10*(*d)[is][n3-7-2][ix ][iz] + C2t_6*((*fm)[ix][iz]*(*d)[is][n3-7+2][ix ][iz]) + \
          C3t_10*(*d)[is][n3-7-3][ix ][iz] + \
          C4t_10*(*d)[is][n3-7-4][ix ][iz] + \
          C5t_10*((*d)[is][n3-7-5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][n3-7][ix ][iz ]  + \
          C1x * ((*d)[is][n3-7][ix + 1 ][iz ] + (*d)[is][n3-7][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][n3-7][ix + 2 ][iz ] + (*d)[is][n3-7][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][n3-7][ix + 3 ][iz ] + (*d)[is][n3-7][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][n3-7][ix + 4 ][iz ] + (*d)[is][n3-7][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][n3-7][ix + 5 ][iz ] + (*d)[is][n3-7][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][n3-7][ix ][iz ]) + \
          C1z * ((*d)[is][n3-7][ix ][iz + 1 ] + (*d)[is][n3-7][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][n3-7][ix ][iz + 2 ] + (*d)[is][n3-7][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][n3-7][ix ][iz + 3 ] + (*d)[is][n3-7][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][n3-7][ix ][iz + 4 ] + (*d)[is][n3-7][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][n3-7][ix ][iz + 5 ] + (*d)[is][n3-7][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][n3-8][ix ][iz ]-(*d)[is][n3-7][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][n3-7][ix ][iz ];

        (*m)[is][n3-5][ix][iz] +=   //second time deriv
          (C0t_8*((*fm)[ix][iz]*(*d)[is][n3-6][ix ][iz]) + \
          C1t_10*(*d)[is][n3-6-1][ix ][iz] + C1t_6*((*fm)[ix][iz]*(*d)[is][n3-6+1][ix ][iz]) + \
          C2t_10*(*d)[is][n3-6-2][ix ][iz] + C2t_4*((*fm)[ix][iz]*(*d)[is][n3-6+2][ix ][iz]) + \
          C3t_10*(*d)[is][n3-6-3][ix ][iz] + \
          C4t_10*(*d)[is][n3-6-4][ix ][iz] + \
          C5t_10*((*d)[is][n3-6-5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][n3-6][ix ][iz ] + \
          C1x * ((*d)[is][n3-6][ix + 1 ][iz ] + (*d)[is][n3-6][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][n3-6][ix + 2 ][iz ] + (*d)[is][n3-6][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][n3-6][ix + 3 ][iz ] + (*d)[is][n3-6][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][n3-6][ix + 4 ][iz ] + (*d)[is][n3-6][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][n3-6][ix + 5 ][iz ] + (*d)[is][n3-6][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][n3-6][ix ][iz ]) + \
          C1z * ((*d)[is][n3-6][ix ][iz + 1 ] + (*d)[is][n3-6][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][n3-6][ix ][iz + 2 ] + (*d)[is][n3-6][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][n3-6][ix ][iz + 3 ] + (*d)[is][n3-6][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][n3-6][ix ][iz + 4 ] + (*d)[is][n3-6][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][n3-6][ix ][iz + 5 ] + (*d)[is][n3-6][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][n3-7][ix ][iz ]-(*d)[is][n3-6][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][n3-6][ix ][iz ];

        (*m)[is][n3-4][ix][iz] +=   //second time deriv
          (C0t_6*((*fm)[ix][iz]*(*d)[is][n3-5][ix ][iz]) + \
          C1t_8*(*d)[is][n3-5-1][ix ][iz] + C1t_4*((*fm)[ix][iz]*(*d)[is][n3-5+1][ix ][iz]) + \
          C2t_10*(*d)[is][n3-5-2][ix ][iz] + \
          C3t_10*(*d)[is][n3-5-3][ix ][iz] + \
          C4t_10*(*d)[is][n3-5-4][ix ][iz] + \
          C5t_10*((*d)[is][n3-5-5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][n3-5][ix ][iz ] + \
          C1x * ((*d)[is][n3-5][ix + 1 ][iz ] + (*d)[is][n3-5][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][n3-5][ix + 2 ][iz ] + (*d)[is][n3-5][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][n3-5][ix + 3 ][iz ] + (*d)[is][n3-5][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][n3-5][ix + 4 ][iz ] + (*d)[is][n3-5][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][n3-5][ix + 5 ][iz ] + (*d)[is][n3-5][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][n3-5][ix ][iz ]) + \
          C1z * ((*d)[is][n3-5][ix ][iz + 1 ] + (*d)[is][n3-5][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][n3-5][ix ][iz + 2 ] + (*d)[is][n3-5][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][n3-5][ix ][iz + 3 ] + (*d)[is][n3-5][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][n3-5][ix ][iz + 4 ] + (*d)[is][n3-5][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][n3-5][ix ][iz + 5 ] + (*d)[is][n3-5][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][n3-6][ix ][iz ]-(*d)[is][n3-5][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][n3-5][ix ][iz ];

        (*m)[is][n3-3][ix][iz] +=   //second time deriv
          (C0t_4*((*fm)[ix][iz]*(*d)[is][n3-4][ix ][iz]) + \
          C1t_6*(*d)[is][n3-4-1][ix ][iz] + C1t_2*((*fm)[ix][iz]*(*d)[is][n3-4+1][ix ][iz]) + \
          C2t_8*(*d)[is][n3-4-2][ix ][iz] + \
          C3t_10*(*d)[is][n3-4-3][ix ][iz] + \
          C4t_10*(*d)[is][n3-4-4][ix ][iz] + \
          C5t_10*((*d)[is][n3-4-5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][n3-4][ix ][iz ] + \
          C1x * ((*d)[is][n3-4][ix + 1 ][iz ] + (*d)[is][n3-4][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][n3-4][ix + 2 ][iz ] + (*d)[is][n3-4][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][n3-4][ix + 3 ][iz ] + (*d)[is][n3-4][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][n3-4][ix + 4 ][iz ] + (*d)[is][n3-4][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][n3-4][ix + 5 ][iz ] + (*d)[is][n3-4][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][n3-4][ix ][iz ]) + \
          C1z * ((*d)[is][n3-4][ix ][iz + 1 ] + (*d)[is][n3-4][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][n3-4][ix ][iz + 2 ] + (*d)[is][n3-4][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][n3-4][ix ][iz + 3 ] + (*d)[is][n3-4][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][n3-4][ix ][iz + 4 ] + (*d)[is][n3-4][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][n3-4][ix ][iz + 5 ] + (*d)[is][n3-4][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][n3-5][ix ][iz ]-(*d)[is][n3-4][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][n3-4][ix ][iz ];

        (*m)[is][n3-2][ix][iz] +=   //second time deriv
          (C0t_2*((*fm)[ix][iz]*(*d)[is][n3-3][ix ][iz]) + \
          C1t_4*(*d)[is][n3-3-1][ix ][iz] + \
          C2t_6*(*d)[is][n3-3-2][ix ][iz] + \
          C3t_8*(*d)[is][n3-3-3][ix ][iz] + \
          C4t_10*(*d)[is][n3-3-4][ix ][iz] + \
          C5t_10*((*d)[is][n3-3-5][ix ][iz]))*(*s)[ix][iz] - \
          //laplacian
          (C0x *(*d)[is][n3-3][ix ][iz ] + \
          C1x * ((*d)[is][n3-3][ix + 1 ][iz ] + (*d)[is][n3-3][ix - 1 ][iz ]) + \
          C2x * ((*d)[is][n3-3][ix + 2 ][iz ] + (*d)[is][n3-3][ix - 2 ][iz ]) + \
          C3x * ((*d)[is][n3-3][ix + 3 ][iz ] + (*d)[is][n3-3][ix - 3 ][iz ]) + \
          C4x * ((*d)[is][n3-3][ix + 4 ][iz ] + (*d)[is][n3-3][ix - 4 ][iz ]) + \
          C5x * ((*d)[is][n3-3][ix + 5 ][iz ] + (*d)[is][n3-3][ix - 5 ][iz ]) + \
          C0z * ((*d)[is][n3-3][ix ][iz ]) + \
          C1z * ((*d)[is][n3-3][ix ][iz + 1 ] + (*d)[is][n3-3][ix ][iz - 1 ]) + \
          C2z * ((*d)[is][n3-3][ix ][iz + 2 ] + (*d)[is][n3-3][ix ][iz - 2 ]) + \
          C3z * ((*d)[is][n3-3][ix ][iz + 3 ] + (*d)[is][n3-3][ix ][iz - 3 ]) + \
          C4z * ((*d)[is][n3-3][ix ][iz + 4 ] + (*d)[is][n3-3][ix ][iz - 4 ]) + \
          C5z * ((*d)[is][n3-3][ix ][iz + 5 ] + (*d)[is][n3-3][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*((*d)[is][n3-4][ix ][iz ]-(*d)[is][n3-3][ix ][iz ]) + \
          //sponge second term
          (*gs)[ix][iz]*(*d)[is][n3-3][ix ][iz ];

        (*m)[is][n3-1][ix][iz] +=   //second time deriv
          (C1t_2*(*d)[is][n3-2-1][ix ][iz] + \
          C2t_4*(*d)[is][n3-2-2][ix ][iz] + \
          C3t_6*(*d)[is][n3-2-3][ix ][iz] + \
          C4t_8*(*d)[is][n3-2-4][ix ][iz] + \
          C5t_10*((*d)[is][n3-2-5][ix ][iz]))*(*s)[ix][iz] + \
          //sponge first term
          (*g)[ix][iz]*(*d)[is][n3-3][ix ][iz ];
      }
    }
  }

  #pragma omp parallel for collapse(4)
  for(int is = 0; is < n4; is++) {
    for (int it = 9; it < n3-11; it++) {
      for (int ix = FAT; ix < n2-FAT; ix++) {
        for (int iz = FAT; iz < n1-FAT; iz++) {
          (*m)[is][it+1][ix][iz] += //second time deriv
            //C0t_10*((*fm)[ix][iz]*(*d)[is][it][ix ][iz])*(*s)[ix][iz] -
            (C0t_10*(*fm)[ix][iz]* (*d)[is][it][ix ][iz ]+ \
            C1t_10*((*d)[is][it-1][ix ][iz ] + (*fm)[ix][iz]*(*d)[is][it + 1][ix ][iz]) + \
            C2t_10*((*d)[is][it-2][ix ][iz ] + (*fm)[ix][iz]*(*d)[is][it + 2][ix ][iz]) + \
            C3t_10*((*d)[is][it-3][ix ][iz ] + (*fm)[ix][iz]*(*d)[is][it + 3][ix ][iz]) + \
            C4t_10*((*d)[is][it-4][ix ][iz ] + (*fm)[ix][iz]*(*d)[is][it + 4][ix ][iz]) + \
            C5t_10*((*d)[is][it-5][ix ][iz ] + (*fm)[ix][iz]*(*d)[is][it + 5][ix ][iz]))*(*s)[ix][iz]  - \
            //laplacian
            (C0x *(*d)[is][it][ix ][iz ] + \
            C1x * ((*d)[is][it][ix + 1 ][iz ] + (*d)[is][it][ix - 1 ][iz ]) + \
            C2x * ((*d)[is][it][ix + 2 ][iz ] + (*d)[is][it][ix - 2 ][iz ]) + \
            C3x * ((*d)[is][it][ix + 3 ][iz ] + (*d)[is][it][ix - 3 ][iz ]) + \
            C4x * ((*d)[is][it][ix + 4 ][iz ] + (*d)[is][it][ix - 4 ][iz ]) + \
            C5x * ((*d)[is][it][ix + 5 ][iz ] + (*d)[is][it][ix - 5 ][iz ]) + \
            C0z * ((*d)[is][it][ix ][iz ]) + \
            C1z * ((*d)[is][it][ix ][iz + 1 ] + (*d)[is][it][ix ][iz - 1 ]) + \
            C2z * ((*d)[is][it][ix ][iz + 2 ] + (*d)[is][it][ix ][iz - 2 ]) + \
            C3z * ((*d)[is][it][ix ][iz + 3 ] + (*d)[is][it][ix ][iz - 3 ]) + \
            C4z * ((*d)[is][it][ix ][iz + 4 ] + (*d)[is][it][ix ][iz - 4 ]) + \
            C5z * ((*d)[is][it][ix ][iz + 5 ] + (*d)[is][it][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
            //sponge first term
            (*g)[ix][iz]*((*d)[is][it-1][ix ][iz ]-(*d)[is][it][ix ][iz ]) + \
            //sponge second term
            (*gs)[ix][iz]*(*d)[is][it][ix ][iz ];
        }
      }
    }
  }
}

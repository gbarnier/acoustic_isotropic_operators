#include <WaveReconV10.h>

WaveReconV10::WaveReconV10(const std::shared_ptr<SEP::float3DReg>model,
                         const std::shared_ptr<SEP::float3DReg>data,
                         const std::shared_ptr<SEP::float2DReg>slsqModel,
                         float                                     U_0,
                         float 					alpha,
			 int					spongeWidth) {
std::cerr << "VERSION 10" << std::endl;
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
void WaveReconV10::set_slsq(std::shared_ptr<SEP::float2DReg>slsq){
	_slsq=slsq;
}

void WaveReconV10::forward(const bool                         add,
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
      (*d)[0][ix][iz] += 0*(*m)[0][ix ][iz ];
      (*d)[1][ix][iz] += 0*(*m)[1][ix ][iz ];
    }
  }
  #pragma omp parallel for collapse(3)
  for (int it = 1; it < n3-1; it++) { //time
    for (int ix = FAT; ix < n2-FAT; ix++) { //x
      for (int iz = FAT; iz < n1-FAT; iz++) { //z
        (*d)[it+1][ix][iz] +=//second time deriv
		//C0t_10*((*fm)[ix][iz] * (*m)[it][ix ][iz ])*(*s)[ix][iz]  -
       		(C0t_2*(*fm)[ix][iz]* (*m)[it][ix ][iz ] + \
                C1t_2*((*fm)[ix][iz] * (*m)[it-1][ix ][iz ]+(*m)[it + 1][ix ][iz]))*(*s)[ix][iz]  - \
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

void WaveReconV10::adjoint(const bool                         add,
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
      (*m)[0][ix][iz] +=  0*(*d)[0][ix][iz] +  C1t_2*((*fm)[ix][iz] * (*d)[2][ix ][iz ])*(*s)[ix][iz];
      (*m)[1][ix][iz] +=  0*(*d)[1][ix][iz] + 
       		(C0t_2*(*fm)[ix][iz]* (*d)[2][ix ][iz ] + \
                C1t_2*((*fm)[ix][iz] * (*d)[2+1][ix ][iz ]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*fm)[ix][iz]*(*d)[2][ix ][iz ] + \
                C1x * ((*fm)[ix+1][iz]*(*d)[2][ix + 1 ][iz ] + (*fm)[ix-1][iz]*(*d)[2][ix - 1 ][iz ]) + \
                C2x * ((*fm)[ix+2][iz]*(*d)[2][ix + 2 ][iz ] + (*fm)[ix-2][iz]*(*d)[2][ix - 2 ][iz ]) + \
                C3x * ((*fm)[ix+3][iz]*(*d)[2][ix + 3 ][iz ] + (*fm)[ix-3][iz]*(*d)[2][ix - 3 ][iz ]) + \
                C4x * ((*fm)[ix+4][iz]*(*d)[2][ix + 4 ][iz ] + (*fm)[ix-4][iz]*(*d)[2][ix - 4 ][iz ]) + \
                C5x * ((*fm)[ix+5][iz]*(*d)[2][ix + 5 ][iz ] + (*fm)[ix-5][iz]*(*d)[2][ix - 5 ][iz ]) + \
                C0z * ((*fm)[ix][iz]*(*d)[2][ix ][iz ]) + \
                C1z * ((*fm)[ix][iz+1]*(*d)[2][ix ][iz + 1 ] + (*fm)[ix][iz-1]*(*d)[2][ix ][iz - 1 ]) + \
                C2z * ((*fm)[ix][iz+2]*(*d)[2][ix ][iz + 2 ] + (*fm)[ix][iz-2]*(*d)[2][ix ][iz - 2 ]) + \
                C3z * ((*fm)[ix][iz+3]*(*d)[2][ix ][iz + 3 ] + (*fm)[ix][iz-3]*(*d)[2][ix ][iz - 3 ]) + \
                C4z * ((*fm)[ix][iz+4]*(*d)[2][ix ][iz + 4 ] + (*fm)[ix][iz-4]*(*d)[2][ix ][iz - 4 ]) + \
                C5z * ((*fm)[ix][iz+5]*(*d)[2][ix ][iz + 5 ] + (*fm)[ix][iz-5]*(*d)[2][ix ][iz - 5 ])) + \
		//sponge first term
		(*g)[ix][iz]*((*d)[0][ix ][iz ]) + \
		//sponge second term
		(*gs)[ix][iz]*(*d)[0][ix ][iz ];
 
      (*m)[n3-2][ix][iz] +=   //second time deriv
			//C0t_2*((*fm)[ix][iz]*(*d)[n3-3][ix ][iz])*(*s)[ix][iz]  -
                         (C0t_2*((*fm)[ix][iz]*(*d)[n3-1][ix ][iz]) + \
                         C1t_2*(*d)[n3-1-1][ix ][iz])*(*s)[ix][iz] - \
			//laplacian
                         (C0x *(*d)[n3-1][ix ][iz ] + \
                         C1x * ((*d)[n3-1][ix + 1 ][iz ] + (*d)[n3-1][ix - 1 ][iz ]) + \
                         C2x * ((*d)[n3-1][ix + 2 ][iz ] + (*d)[n3-1][ix - 2 ][iz ]) + \
                         C3x * ((*d)[n3-1][ix + 3 ][iz ] + (*d)[n3-1][ix - 3 ][iz ]) + \
                         C4x * ((*d)[n3-1][ix + 4 ][iz ] + (*d)[n3-1][ix - 4 ][iz ]) + \
                         C5x * ((*d)[n3-1][ix + 5 ][iz ] + (*d)[n3-1][ix - 5 ][iz ]) + \
                         C0z * ((*d)[n3-1][ix ][iz ]) + \
                         C1z * ((*d)[n3-1][ix ][iz + 1 ] + (*d)[n3-1][ix ][iz - 1 ]) + \
                         C2z * ((*d)[n3-1][ix ][iz + 2 ] + (*d)[n3-1][ix ][iz - 2 ]) + \
                         C3z * ((*d)[n3-1][ix ][iz + 3 ] + (*d)[n3-1][ix ][iz - 3 ]) + \
                         C4z * ((*d)[n3-1][ix ][iz + 4 ] + (*d)[n3-1][ix ][iz - 4 ]) + \
                         C5z * ((*d)[n3-1][ix ][iz + 5 ] + (*d)[n3-1][ix ][iz - 5 ]))*(*fm)[ix][iz] + \
		         //sponge first term
		         (*g)[ix][iz]*((*d)[n3-4][ix ][iz ]-(*d)[n3-3][ix ][iz ]) + \
		         //sponge second term
		         (*gs)[ix][iz]*(*d)[n3-3][ix ][iz ];
      (*m)[n3-1][ix][iz] +=   (C1t_2*(*d)[n3-1][ix ][iz])*(*s)[ix][iz];
    }
  }

  #pragma omp parallel for collapse(3)
  for (int it = 3; it < n3-1; it++) {
    for (int ix = FAT; ix < n2-FAT; ix++) {
      for (int iz = FAT; iz < n1-FAT; iz++) {
        (*m)[it-1][ix][iz] += //second time deriv
                         //C0t_10*((*fm)[ix][iz]*(*d)[it][ix ][iz])*(*s)[ix][iz] - 
       			 (C0t_2*(*fm)[ix][iz]* (*d)[it][ix ][iz ]+ \
                         C1t_2*((*d)[it-1][ix ][iz ] + (*fm)[ix][iz]*(*d)[it + 1][ix ][iz]))*(*s)[ix][iz]  - \
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




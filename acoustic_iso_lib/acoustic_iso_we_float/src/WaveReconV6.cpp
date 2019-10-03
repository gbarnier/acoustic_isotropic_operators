#include <WaveReconV6.h>

WaveReconV6::WaveReconV6(const std::shared_ptr<SEP::float3DReg>model,
                         const std::shared_ptr<SEP::float3DReg>data,
                         const std::shared_ptr<SEP::float2DReg>slsqModel,
                         int                                    boundaryCond,
			 int					spongeWidth) {
std::cerr << "VERSION 6" << std::endl;
  // ensure model and data dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  // ensure velModel matches spatial dimensions of model and data
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
  // get spatial sampling
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

  C0t = -2.927222222 / (_dt * _dt);
  C1t =  1.666666667 / (_dt * _dt);
  C2t = -0.23809524 / (_dt * _dt);
  C3t =  0.03968254 / (_dt * _dt);
  C4t = -0.00496032 / (_dt * _dt);
  C5t =  0.00031746 / (_dt * _dt);

  setDomainRange(model, data);

    _sponge.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
                                      data->getHyper()->getAxis(2).n));
  
    _sponge->set(0);
    int nz=data->getHyper()->getAxis(1).n;
    int nx=data->getHyper()->getAxis(2).n;
    int nt=data->getHyper()->getAxis(3).n;
    float alphaCos= 0.99;
    for (int ix = FAT; ix < nx-FAT; ix++){
      for (int iz = FAT; iz < nz-FAT; iz++) {
	int distToEdge = std::min(std::min(ix-FAT,iz-FAT),std::min(nz-FAT-iz-1,nx-FAT-ix-1));
	if(distToEdge < _spongeWidth){
          float arg = M_PI / (1.0 * _spongeWidth) * 1.0 * (_spongeWidth-distToEdge);
          float coeff = alphaCos + (1.0-alphaCos) * cos(arg);
	  (*_sponge->_mat)[ix][iz] = coeff; 
	}
	else{
	  (*_sponge->_mat)[ix][iz] = 1; 
	}
      }
    }
//	_sponge->set(1);
//   int iz=nz/2;
//   for (int ix = 0; ix < nx; ix++){
//        //std::cerr << "sponge[" << ix << "][3]=" << (*_sponge->_mat)[ix][3] << std::endl;
//        std::cerr << "sponge[" << ix << "][" << iz << "]=" << (*_sponge->_mat)[ix][iz] << std::endl;
//   }
//   int ix=nx/2;
//   for (int iz = 0; iz < nz; iz++){
//        //std::cerr << "sponge[3][" << iz << "]=" << (*_sponge->_mat)[3][iz] << std::endl;
//        std::cerr << "sponge[" << ix << "][" << iz << "]=" << (*_sponge->_mat)[ix][iz] << std::endl;
//   }
  // set slowness
  _slsq=slsqModel;
 // _slsq_dt2 = std::make_shared<float2DReg>(slsqModel->getHyper()->getAxis(1), slsqModel->getHyper()->getAxis(2));
 // for (int ix = 0; ix < slsqModel->getHyper()->getAxis(2).n; ix++){
 //       	for (int iz = 0; iz < slsqModel->getHyper()->getAxis(1).n; iz++) {
 //       		(*_slsq_dt2->_mat)[ix][iz] = (*slsqModel->_mat)[ix][iz]/_dt2;
 //       	}
 // }
}

void WaveReconV6::forward(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const {
  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);


    const std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
     std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
     std::shared_ptr<float2D> sp = _sponge->_mat;
      std::shared_ptr<float2D> s =
       ((std::dynamic_pointer_cast<float2DReg>(_slsq))->_mat);

  //boundary condition
  #pragma omp parallel for collapse(2)
  for (int ix = FAT; ix < n2-FAT; ix++) { //x
    for (int iz = FAT; iz < n1-FAT; iz++) { //z
      (*d)[n3-10][ix][iz] += //second time deriv
       		(C0t*(*sp)[ix][iz]* (*m)[n3-5][ix ][iz ]+ \
                C1t*((*sp)[ix][iz] * (*m)[n3-5-1][ix ][iz ]+(*m)[n3-5 + 1][ix ][iz]) + \
                C2t*((*sp)[ix][iz] * (*m)[n3-5-2][ix ][iz ]+(*m)[n3-5 + 2][ix ][iz]) + \
                C3t*((*sp)[ix][iz] * (*m)[n3-5-3][ix ][iz ]+(*m)[n3-5 + 3][ix ][iz]) + \
                C4t*((*sp)[ix][iz] * (*m)[n3-5-4][ix ][iz ]+(*m)[n3-5 + 4][ix ][iz]) + \
                C5t*((*sp)[ix][iz] * (*m)[n3-5-5][ix ][iz ]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*sp)[ix][iz]*(*m)[n3-5][ix ][iz ] + \
                C1x * ((*sp)[ix+1][iz]*(*m)[n3-5][ix + 1 ][iz ] + (*sp)[ix-1][iz]*(*m)[n3-5][ix - 1 ][iz ]) + \
                C2x * ((*sp)[ix+2][iz]*(*m)[n3-5][ix + 2 ][iz ] + (*sp)[ix-2][iz]*(*m)[n3-5][ix - 2 ][iz ]) + \
                C3x * ((*sp)[ix+3][iz]*(*m)[n3-5][ix + 3 ][iz ] + (*sp)[ix-3][iz]*(*m)[n3-5][ix - 3 ][iz ]) + \
                C4x * ((*sp)[ix+4][iz]*(*m)[n3-5][ix + 4 ][iz ] + (*sp)[ix-4][iz]*(*m)[n3-5][ix - 4 ][iz ]) + \
                C5x * ((*sp)[ix+5][iz]*(*m)[n3-5][ix + 5 ][iz ] + (*sp)[ix-5][iz]*(*m)[n3-5][ix - 5 ][iz ]) + \
                C0z * ((*sp)[ix][iz]*(*m)[n3-5][ix ][iz ]) + \
                C1z * ((*sp)[ix][iz+1]*(*m)[n3-5][ix ][iz + 1 ] + (*sp)[ix][iz-1]*(*m)[n3-5][ix ][iz - 1 ]) + \
                C2z * ((*sp)[ix][iz+2]*(*m)[n3-5][ix ][iz + 2 ] + (*sp)[ix][iz-2]*(*m)[n3-5][ix ][iz - 2 ]) + \
                C3z * ((*sp)[ix][iz+3]*(*m)[n3-5][ix ][iz + 3 ] + (*sp)[ix][iz-3]*(*m)[n3-5][ix ][iz - 3 ]) + \
                C4z * ((*sp)[ix][iz+4]*(*m)[n3-5][ix ][iz + 4 ] + (*sp)[ix][iz-4]*(*m)[n3-5][ix ][iz - 4 ]) + \
                C5z * ((*sp)[ix][iz+5]*(*m)[n3-5][ix ][iz + 5 ] + (*sp)[ix][iz-5]*(*m)[n3-5][ix ][iz - 5 ]));
      (*d)[n3-9][ix][iz] += //second time deriv
       		(C0t*(*sp)[ix][iz]* (*m)[n3-4][ix ][iz ]+ \
                C1t*((*sp)[ix][iz] * (*m)[n3-4-1][ix ][iz ]+(*m)[n3-4 + 1][ix ][iz]) + \
                C2t*((*sp)[ix][iz] * (*m)[n3-4-2][ix ][iz ]+(*m)[n3-4 + 2][ix ][iz]) + \
                C3t*((*sp)[ix][iz] * (*m)[n3-4-3][ix ][iz ]+(*m)[n3-4 + 3][ix ][iz]) + \
                C4t*((*sp)[ix][iz] * (*m)[n3-4-4][ix ][iz ]) + \
                C5t*((*sp)[ix][iz] * (*m)[n3-4-5][ix ][iz ]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*sp)[ix][iz]*(*m)[n3-4][ix ][iz ] + \
                C1x * ((*sp)[ix+1][iz]*(*m)[n3-4][ix + 1 ][iz ] + (*sp)[ix-1][iz]*(*m)[n3-4][ix - 1 ][iz ]) + \
                C2x * ((*sp)[ix+2][iz]*(*m)[n3-4][ix + 2 ][iz ] + (*sp)[ix-2][iz]*(*m)[n3-4][ix - 2 ][iz ]) + \
                C3x * ((*sp)[ix+3][iz]*(*m)[n3-4][ix + 3 ][iz ] + (*sp)[ix-3][iz]*(*m)[n3-4][ix - 3 ][iz ]) + \
                C4x * ((*sp)[ix+4][iz]*(*m)[n3-4][ix + 4 ][iz ] + (*sp)[ix-4][iz]*(*m)[n3-4][ix - 4 ][iz ]) + \
                C5x * ((*sp)[ix+5][iz]*(*m)[n3-4][ix + 5 ][iz ] + (*sp)[ix-5][iz]*(*m)[n3-4][ix - 5 ][iz ]) + \
                C0z * ((*sp)[ix][iz]*(*m)[n3-4][ix ][iz ]) + \
                C1z * ((*sp)[ix][iz+1]*(*m)[n3-4][ix ][iz + 1 ] + (*sp)[ix][iz-1]*(*m)[n3-4][ix ][iz - 1 ]) + \
                C2z * ((*sp)[ix][iz+2]*(*m)[n3-4][ix ][iz + 2 ] + (*sp)[ix][iz-2]*(*m)[n3-4][ix ][iz - 2 ]) + \
                C3z * ((*sp)[ix][iz+3]*(*m)[n3-4][ix ][iz + 3 ] + (*sp)[ix][iz-3]*(*m)[n3-4][ix ][iz - 3 ]) + \
                C4z * ((*sp)[ix][iz+4]*(*m)[n3-4][ix ][iz + 4 ] + (*sp)[ix][iz-4]*(*m)[n3-4][ix ][iz - 4 ]) + \
                C5z * ((*sp)[ix][iz+5]*(*m)[n3-4][ix ][iz + 5 ] + (*sp)[ix][iz-5]*(*m)[n3-4][ix ][iz - 5 ]));
      (*d)[n3-8][ix][iz] += //second time deriv
       		(C0t*(*sp)[ix][iz]* (*m)[n3-3][ix ][iz ]+ \
                C1t*((*sp)[ix][iz] * (*m)[n3-3-1][ix ][iz ]+(*m)[n3-3 + 1][ix ][iz]) + \
                C2t*((*sp)[ix][iz] * (*m)[n3-3-2][ix ][iz ]+(*m)[n3-3 + 2][ix ][iz]) + \
                C3t*((*sp)[ix][iz] * (*m)[n3-3-3][ix ][iz ]) + \
                C4t*((*sp)[ix][iz] * (*m)[n3-3-4][ix ][iz ]) + \
                C5t*((*sp)[ix][iz] * (*m)[n3-3-5][ix ][iz ]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*sp)[ix][iz]*(*m)[n3-3][ix ][iz ] + \
                C1x * ((*sp)[ix+1][iz]*(*m)[n3-3][ix + 1 ][iz ] + (*sp)[ix-1][iz]*(*m)[n3-3][ix - 1 ][iz ]) + \
                C2x * ((*sp)[ix+2][iz]*(*m)[n3-3][ix + 2 ][iz ] + (*sp)[ix-2][iz]*(*m)[n3-3][ix - 2 ][iz ]) + \
                C3x * ((*sp)[ix+3][iz]*(*m)[n3-3][ix + 3 ][iz ] + (*sp)[ix-3][iz]*(*m)[n3-3][ix - 3 ][iz ]) + \
                C4x * ((*sp)[ix+4][iz]*(*m)[n3-3][ix + 4 ][iz ] + (*sp)[ix-4][iz]*(*m)[n3-3][ix - 4 ][iz ]) + \
                C5x * ((*sp)[ix+5][iz]*(*m)[n3-3][ix + 5 ][iz ] + (*sp)[ix-5][iz]*(*m)[n3-3][ix - 5 ][iz ]) + \
                C0z * ((*sp)[ix][iz]*(*m)[n3-3][ix ][iz ]) + \
                C1z * ((*sp)[ix][iz+1]*(*m)[n3-3][ix ][iz + 1 ] + (*sp)[ix][iz-1]*(*m)[n3-3][ix ][iz - 1 ]) + \
                C2z * ((*sp)[ix][iz+2]*(*m)[n3-3][ix ][iz + 2 ] + (*sp)[ix][iz-2]*(*m)[n3-3][ix ][iz - 2 ]) + \
                C3z * ((*sp)[ix][iz+3]*(*m)[n3-3][ix ][iz + 3 ] + (*sp)[ix][iz-3]*(*m)[n3-3][ix ][iz - 3 ]) + \
                C4z * ((*sp)[ix][iz+4]*(*m)[n3-3][ix ][iz + 4 ] + (*sp)[ix][iz-4]*(*m)[n3-3][ix ][iz - 4 ]) + \
                C5z * ((*sp)[ix][iz+5]*(*m)[n3-3][ix ][iz + 5 ] + (*sp)[ix][iz-5]*(*m)[n3-3][ix ][iz - 5 ]));
      (*d)[n3-7][ix][iz] += //second time deriv
       		(C0t*(*sp)[ix][iz]* (*m)[n3-2][ix ][iz ]+ \
                C1t*((*sp)[ix][iz] * (*m)[n3-2-1][ix ][iz ]+(*m)[n3-2 + 1][ix ][iz]) + \
                C2t*((*sp)[ix][iz] * (*m)[n3-2-2][ix ][iz ]) + \
                C3t*((*sp)[ix][iz] * (*m)[n3-2-3][ix ][iz ]) + \
                C4t*((*sp)[ix][iz] * (*m)[n3-2-4][ix ][iz ]) + \
                C5t*((*sp)[ix][iz] * (*m)[n3-2-5][ix ][iz ]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*sp)[ix][iz]*(*m)[n3-2][ix ][iz ] + \
                C1x * ((*sp)[ix+1][iz]*(*m)[n3-2][ix + 1 ][iz ] + (*sp)[ix-1][iz]*(*m)[n3-2][ix - 1 ][iz ]) + \
                C2x * ((*sp)[ix+2][iz]*(*m)[n3-2][ix + 2 ][iz ] + (*sp)[ix-2][iz]*(*m)[n3-2][ix - 2 ][iz ]) + \
                C3x * ((*sp)[ix+3][iz]*(*m)[n3-2][ix + 3 ][iz ] + (*sp)[ix-3][iz]*(*m)[n3-2][ix - 3 ][iz ]) + \
                C4x * ((*sp)[ix+4][iz]*(*m)[n3-2][ix + 4 ][iz ] + (*sp)[ix-4][iz]*(*m)[n3-2][ix - 4 ][iz ]) + \
                C5x * ((*sp)[ix+5][iz]*(*m)[n3-2][ix + 5 ][iz ] + (*sp)[ix-5][iz]*(*m)[n3-2][ix - 5 ][iz ]) + \
                C0z * ((*sp)[ix][iz]*(*m)[n3-2][ix ][iz ]) + \
                C1z * ((*sp)[ix][iz+1]*(*m)[n3-2][ix ][iz + 1 ] + (*sp)[ix][iz-1]*(*m)[n3-2][ix ][iz - 1 ]) + \
                C2z * ((*sp)[ix][iz+2]*(*m)[n3-2][ix ][iz + 2 ] + (*sp)[ix][iz-2]*(*m)[n3-2][ix ][iz - 2 ]) + \
                C3z * ((*sp)[ix][iz+3]*(*m)[n3-2][ix ][iz + 3 ] + (*sp)[ix][iz-3]*(*m)[n3-2][ix ][iz - 3 ]) + \
                C4z * ((*sp)[ix][iz+4]*(*m)[n3-2][ix ][iz + 4 ] + (*sp)[ix][iz-4]*(*m)[n3-2][ix ][iz - 4 ]) + \
                C5z * ((*sp)[ix][iz+5]*(*m)[n3-2][ix ][iz + 5 ] + (*sp)[ix][iz-5]*(*m)[n3-2][ix ][iz - 5 ]));
      (*d)[n3-6][ix][iz] += //second time deriv
       		(C0t*(*sp)[ix][iz]* (*m)[n3-1][ix ][iz ]+ \
                C1t*((*sp)[ix][iz] * (*m)[n3-1-1][ix ][iz ]) + \
                C2t*((*sp)[ix][iz] * (*m)[n3-1-2][ix ][iz ]) + \
                C3t*((*sp)[ix][iz] * (*m)[n3-1-3][ix ][iz ]) + \
                C4t*((*sp)[ix][iz] * (*m)[n3-1-4][ix ][iz ]) + \
                C5t*((*sp)[ix][iz] * (*m)[n3-1-5][ix ][iz ]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*sp)[ix][iz]*(*m)[n3-1][ix ][iz ] + \
                C1x * ((*sp)[ix+1][iz]*(*m)[n3-1][ix + 1 ][iz ] + (*sp)[ix-1][iz]*(*m)[n3-1][ix - 1 ][iz ]) + \
                C2x * ((*sp)[ix+2][iz]*(*m)[n3-1][ix + 2 ][iz ] + (*sp)[ix-2][iz]*(*m)[n3-1][ix - 2 ][iz ]) + \
                C3x * ((*sp)[ix+3][iz]*(*m)[n3-1][ix + 3 ][iz ] + (*sp)[ix-3][iz]*(*m)[n3-1][ix - 3 ][iz ]) + \
                C4x * ((*sp)[ix+4][iz]*(*m)[n3-1][ix + 4 ][iz ] + (*sp)[ix-4][iz]*(*m)[n3-1][ix - 4 ][iz ]) + \
                C5x * ((*sp)[ix+5][iz]*(*m)[n3-1][ix + 5 ][iz ] + (*sp)[ix-5][iz]*(*m)[n3-1][ix - 5 ][iz ]) + \
                C0z * ((*sp)[ix][iz]*(*m)[n3-1][ix ][iz ]) + \
                C1z * ((*sp)[ix][iz+1]*(*m)[n3-1][ix ][iz + 1 ] + (*sp)[ix][iz-1]*(*m)[n3-1][ix ][iz - 1 ]) + \
                C2z * ((*sp)[ix][iz+2]*(*m)[n3-1][ix ][iz + 2 ] + (*sp)[ix][iz-2]*(*m)[n3-1][ix ][iz - 2 ]) + \
                C3z * ((*sp)[ix][iz+3]*(*m)[n3-1][ix ][iz + 3 ] + (*sp)[ix][iz-3]*(*m)[n3-1][ix ][iz - 3 ]) + \
                C4z * ((*sp)[ix][iz+4]*(*m)[n3-1][ix ][iz + 4 ] + (*sp)[ix][iz-4]*(*m)[n3-1][ix ][iz - 4 ]) + \
                C5z * ((*sp)[ix][iz+5]*(*m)[n3-1][ix ][iz + 5 ] + (*sp)[ix][iz-5]*(*m)[n3-1][ix ][iz - 5 ]));
      (*d)[n3-5][ix][iz] += (*m)[0 ][ix ][iz];
      (*d)[n3-4][ix][iz] += (*m)[1 ][ix ][iz];
      (*d)[n3-3][ix][iz] += (*m)[2 ][ix ][iz];
      (*d)[n3-2][ix][iz] += (*m)[3 ][ix ][iz];
      (*d)[n3-1][ix][iz] += (*m)[4 ][ix ][iz];
    }
  }
  #pragma omp parallel for collapse(3)
  for (int it = 5; it < n3-5; it++) { //time
    for (int ix = FAT; ix < n2-FAT; ix++) { //x
      for (int iz = FAT; iz < n1-FAT; iz++) { //z
        (*d)[it-5][ix][iz] +=//second time deriv
       		(C0t*(*sp)[ix][iz]* (*m)[it][ix ][iz ]+ \
                C1t*((*sp)[ix][iz] * (*m)[it-1][ix ][iz ]+(*m)[it + 1][ix ][iz]) + \
                C2t*((*sp)[ix][iz] * (*m)[it-2][ix ][iz ]+(*m)[it + 2][ix ][iz]) + \
                C3t*((*sp)[ix][iz] * (*m)[it-3][ix ][iz ]+(*m)[it + 3][ix ][iz]) + \
                C4t*((*sp)[ix][iz] * (*m)[it-4][ix ][iz ]+(*m)[it + 4][ix ][iz]) + \
                C5t*((*sp)[ix][iz] * (*m)[it-5][ix ][iz ]+(*m)[it + 5][ix ][iz]))*(*s)[ix][iz]  - \
                //laplacian
                (C0x * (*sp)[ix][iz]*(*m)[it][ix ][iz ] + \
                C1x * ((*sp)[ix+1][iz]*(*m)[it][ix + 1 ][iz ] + (*sp)[ix-1][iz]*(*m)[it][ix - 1 ][iz ]) + \
                C2x * ((*sp)[ix+2][iz]*(*m)[it][ix + 2 ][iz ] + (*sp)[ix-2][iz]*(*m)[it][ix - 2 ][iz ]) + \
                C3x * ((*sp)[ix+3][iz]*(*m)[it][ix + 3 ][iz ] + (*sp)[ix-3][iz]*(*m)[it][ix - 3 ][iz ]) + \
                C4x * ((*sp)[ix+4][iz]*(*m)[it][ix + 4 ][iz ] + (*sp)[ix-4][iz]*(*m)[it][ix - 4 ][iz ]) + \
                C5x * ((*sp)[ix+5][iz]*(*m)[it][ix + 5 ][iz ] + (*sp)[ix-5][iz]*(*m)[it][ix - 5 ][iz ]) + \
                C0z * ((*sp)[ix][iz]*(*m)[it][ix ][iz ]) + \
                C1z * ((*sp)[ix][iz+1]*(*m)[it][ix ][iz + 1 ] + (*sp)[ix][iz-1]*(*m)[it][ix ][iz - 1 ]) + \
                C2z * ((*sp)[ix][iz+2]*(*m)[it][ix ][iz + 2 ] + (*sp)[ix][iz-2]*(*m)[it][ix ][iz - 2 ]) + \
                C3z * ((*sp)[ix][iz+3]*(*m)[it][ix ][iz + 3 ] + (*sp)[ix][iz-3]*(*m)[it][ix ][iz - 3 ]) + \
                C4z * ((*sp)[ix][iz+4]*(*m)[it][ix ][iz + 4 ] + (*sp)[ix][iz-4]*(*m)[it][ix ][iz - 4 ]) + \
                C5z * ((*sp)[ix][iz+5]*(*m)[it][ix ][iz + 5 ] + (*sp)[ix][iz-5]*(*m)[it][ix ][iz - 5 ]));
      }
    }
  }

}

void WaveReconV6::adjoint(const bool                         add,
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
     std::shared_ptr<float2D> sp = _sponge->_mat;


  #pragma omp parallel for collapse(2)
  for (int ix = FAT; ix < n2-FAT; ix++) { //x
    for (int iz = FAT; iz < n1-FAT; iz++) { //z
      (*m)[0][ix][iz] +=   //second time deriv
                         (C5t*((*sp)[ix][iz]*(*d)[0][ix ][iz]))*(*s)[ix][iz] + \
                         //initial condition 
       			 (*d)[n3-5][ix ][iz ];
      (*m)[1][ix][iz] +=   //second time deriv
                         (C4t*((*sp)[ix][iz]*(*d)[0][ix ][iz]) + \
                         C5t*((*sp)[ix][iz]*(*d)[1][ix ][iz]))*(*s)[ix][iz] + \
                         //initial condition 
       			 (*d)[n3-4][ix ][iz ];
      (*m)[2][ix][iz] +=   //second time deriv
                         (C3t*((*sp)[ix][iz]*(*d)[0][ix ][iz]) + \
                         C4t*((*sp)[ix][iz]*(*d)[1][ix ][iz]) + \
                         C5t*((*sp)[ix][iz]*(*d)[2][ix ][iz]))*(*s)[ix][iz] + \
                         //initial condition 
       			 (*d)[n3-3][ix ][iz ];
      (*m)[3][ix][iz] +=   //second time deriv
                         (C2t*((*sp)[ix][iz]*(*d)[0][ix ][iz]) + \
                         C3t*((*sp)[ix][iz]*(*d)[1][ix ][iz]) + \
                         C4t*((*sp)[ix][iz]*(*d)[2][ix ][iz]) + \
                         C5t*((*sp)[ix][iz]*(*d)[3][ix ][iz]))*(*s)[ix][iz] + \
                         //initial condition 
       			 (*d)[n3-2][ix ][iz ];
      (*m)[4][ix][iz] +=   //second time deriv
                         (C1t*((*sp)[ix][iz]*(*d)[0][ix ][iz]) + \
                         C2t*((*sp)[ix][iz]*(*d)[1][ix ][iz]) + \
                         C3t*((*sp)[ix][iz]*(*d)[2][ix ][iz]) + \
                         C4t*((*sp)[ix][iz]*(*d)[3][ix ][iz]) + \
                         C5t*((*sp)[ix][iz]*(*d)[4][ix ][iz]))*(*s)[ix][iz]+ \
                         //initial condition 
       			 (*d)[n3-1][ix ][iz ];
      (*m)[5][ix][iz] +=   //second time deriv
       			 (C0t* (*sp)[ix][iz]*(*d)[0][ix ][iz ]+ \
                         C1t*((*sp)[ix][iz]*(*d)[1][ix ][iz]) + \
                         C2t*((*sp)[ix][iz]*(*d)[2][ix ][iz]) + \
                         C3t*((*sp)[ix][iz]*(*d)[3][ix ][iz]) + \
                         C4t*((*sp)[ix][iz]*(*d)[4][ix ][iz]) + \
                         C5t*((*sp)[ix][iz]*(*d)[5][ix ][iz]))*(*s)[ix][iz]  - \
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
                         C5z * ((*d)[0][ix ][iz + 5 ] + (*d)[0][ix ][iz - 5 ]))*(*sp)[ix][iz];
      (*m)[6][ix][iz] +=   //second time deriv
       			 (C0t* (*sp)[ix][iz]*(*d)[1][ix ][iz ]+ \
                         C1t*((*d)[0][ix ][iz ] + (*sp)[ix][iz]*(*d)[2][ix ][iz]) + \
                         C2t*((*sp)[ix][iz]*(*d)[3][ix ][iz]) + \
                         C3t*((*sp)[ix][iz]*(*d)[4][ix ][iz]) + \
                         C4t*((*sp)[ix][iz]*(*d)[5][ix ][iz]) + \
                         C5t*((*sp)[ix][iz]*(*d)[6][ix ][iz]))*(*s)[ix][iz]  - \
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
                         C5z * ((*d)[1][ix ][iz + 5 ] + (*d)[1][ix ][iz - 5 ]))*(*sp)[ix][iz];
      (*m)[7][ix][iz] +=   //second time deriv
       			 (C0t* (*sp)[ix][iz]*(*d)[2][ix ][iz ]+ \
                         C1t*((*d)[1][ix ][iz ] + (*sp)[ix][iz]*(*d)[3][ix ][iz]) + \
                         C2t*((*d)[0][ix ][iz ] + (*sp)[ix][iz]*(*d)[4][ix ][iz]) + \
                         C3t*((*sp)[ix][iz]*(*d)[5][ix ][iz]) + \
                         C4t*((*sp)[ix][iz]*(*d)[6][ix ][iz]) + \
                         C5t*((*sp)[ix][iz]*(*d)[7][ix ][iz]))*(*s)[ix][iz]  - \
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
                         C5z * ((*d)[2][ix ][iz + 5 ] + (*d)[2][ix ][iz - 5 ]))*(*sp)[ix][iz];
      (*m)[8][ix][iz] +=   //second time deriv
       			 (C0t* (*sp)[ix][iz]*(*d)[3][ix ][iz ]+ \
                         C1t*((*d)[2][ix ][iz ] + (*sp)[ix][iz]*(*d)[4][ix ][iz]) + \
                         C2t*((*d)[1][ix ][iz ] + (*sp)[ix][iz]*(*d)[5][ix ][iz]) + \
                         C3t*((*d)[0][ix ][iz ] + (*sp)[ix][iz]*(*d)[6][ix ][iz]) + \
                         C4t*((*sp)[ix][iz]*(*d)[7][ix ][iz]) + \
                         C5t*((*sp)[ix][iz]*(*d)[8][ix ][iz]))*(*s)[ix][iz]  - \
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
                         C5z * ((*d)[3][ix ][iz + 5 ] + (*d)[3][ix ][iz - 5 ]))*(*sp)[ix][iz];
      (*m)[9][ix][iz] +=  //second time deriv
       			 (C0t* (*sp)[ix][iz]*(*d)[4][ix ][iz ]+ \
                         C1t*((*d)[3][ix ][iz ] + (*sp)[ix][iz]*(*d)[5][ix ][iz]) + \
                         C2t*((*d)[2][ix ][iz ] + (*sp)[ix][iz]*(*d)[6][ix ][iz]) + \
                         C3t*((*d)[1][ix ][iz ] + (*sp)[ix][iz]*(*d)[7][ix ][iz]) + \
                         C4t*((*d)[0][ix ][iz ] + (*sp)[ix][iz]*(*d)[8][ix ][iz]) + \
                         C5t*((*sp)[ix][iz]*(*d)[9][ix ][iz]))*(*s)[ix][iz]  - \
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
                         C5z * ((*d)[4][ix ][iz + 5 ] + (*d)[4][ix ][iz - 5 ]))*(*sp)[ix][iz];
      (*m)[n3-5][ix][iz] +=   //second time deriv
       			 (C0t*(*sp)[ix][iz]*(*d)[n3-10][ix ][iz ]+ \
                         C1t*((*d)[n3-10-1][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-10 + 1][ix ][iz]) + \
                         C2t*((*d)[n3-10-2][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-10 + 2][ix ][iz]) + \
                         C3t*((*d)[n3-10-3][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-10 + 3][ix ][iz]) + \
                         C4t*((*d)[n3-10-4][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-10 + 4][ix ][iz]) + \
                         C5t*((*d)[n3-10-5][ix ][iz ]))*(*s)[ix][iz]  - \
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
                         C5z * ((*d)[n3-10][ix ][iz + 5 ] + (*d)[n3-10][ix ][iz - 5 ]))*(*sp)[ix][iz];
      (*m)[n3-4][ix][iz] +=   //second time deriv
       			 (C0t* (*sp)[ix][iz]*(*d)[n3-9][ix ][iz ]+ \
                         C1t*((*d)[n3-9-1][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-9 + 1][ix ][iz]) + \
                         C2t*((*d)[n3-9-2][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-9 + 2][ix ][iz]) + \
                         C3t*((*d)[n3-9-3][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-9 + 3][ix ][iz]) + \
                         C4t*((*d)[n3-9-4][ix ][iz ]) + \
                         C5t*((*d)[n3-9-5][ix ][iz ]))*(*s)[ix][iz]  - \
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
                         C5z * ((*d)[n3-9][ix ][iz + 5 ] + (*d)[n3-9][ix ][iz - 5 ]))*(*sp)[ix][iz];
      (*m)[n3-3][ix][iz] +=   //second time deriv
       			 (C0t* (*sp)[ix][iz]*(*d)[n3-8][ix ][iz ]+ \
                         C1t*((*d)[n3-8-1][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-8 + 1][ix ][iz]) + \
                         C2t*((*d)[n3-8-2][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-8 + 2][ix ][iz]) + \
                         C3t*((*d)[n3-8-3][ix ][iz ]) + \
                         C4t*((*d)[n3-8-4][ix ][iz ]) + \
                         C5t*((*d)[n3-8-5][ix ][iz ]))*(*s)[ix][iz]  - \
                         //laplacian
                         (C0x *(*d)[n3-8][ix ][iz ] + \
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
                         C5z * ((*d)[n3-8][ix ][iz + 5 ] + (*d)[n3-8][ix ][iz - 5 ]))*(*sp)[ix][iz];
      (*m)[n3-2][ix][iz] +=   //second time deriv
       			 (C0t* (*sp)[ix][iz]*(*d)[n3-7][ix ][iz ]+ \
                         C1t*((*d)[n3-7-1][ix ][iz ] + (*sp)[ix][iz]*(*d)[n3-7 + 1][ix ][iz]) + \
                         C2t*((*d)[n3-7-2][ix ][iz ]) + \
                         C3t*((*d)[n3-7-3][ix ][iz ]) + \
                         C4t*((*d)[n3-7-4][ix ][iz ]) + \
                         C5t*((*d)[n3-7-5][ix ][iz ]))*(*s)[ix][iz]  - \
                         //laplacian
                         (C0x *(*d)[n3-7][ix ][iz ] + \
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
                         C5z * ((*d)[n3-7][ix ][iz + 5 ] + (*d)[n3-7][ix ][iz - 5 ]))*(*sp)[ix][iz];
      (*m)[n3-1][ix][iz] +=   //second time deriv
       			 (C0t* (*sp)[ix][iz]*(*d)[n3-6][ix ][iz ]+ \
                         C1t*((*d)[n3-6-1][ix ][iz ]) + \
                         C2t*((*d)[n3-6-2][ix ][iz ]) + \
                         C3t*((*d)[n3-6-3][ix ][iz ]) + \
                         C4t*((*d)[n3-6-4][ix ][iz ]) + \
                         C5t*((*d)[n3-6-5][ix ][iz ]))*(*s)[ix][iz]  - \
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
                         C5z * ((*d)[n3-6][ix ][iz + 5 ] + (*d)[n3-6][ix ][iz - 5 ]))*(*sp)[ix][iz];
    }
  }

  #pragma omp parallel for collapse(3)
  for (int it = 5; it < n3-10; it++) {
    for (int ix = FAT; ix < n2-FAT; ix++) {
      for (int iz = FAT; iz < n1-FAT; iz++) {
        (*m)[it+5][ix][iz] += //second time deriv
       			 (C0t*(*sp)[ix][iz]* (*d)[it][ix ][iz ]+ \
                         C1t*((*d)[it-1][ix ][iz ] + (*sp)[ix][iz]*(*d)[it + 1][ix ][iz]) + \
                         C2t*((*d)[it-2][ix ][iz ] + (*sp)[ix][iz]*(*d)[it + 2][ix ][iz]) + \
                         C3t*((*d)[it-3][ix ][iz ] + (*sp)[ix][iz]*(*d)[it + 3][ix ][iz]) + \
                         C4t*((*d)[it-4][ix ][iz ] + (*sp)[ix][iz]*(*d)[it + 4][ix ][iz]) + \
                         C5t*((*d)[it-5][ix ][iz ] + (*sp)[ix][iz]*(*d)[it + 5][ix ][iz]))*(*s)[ix][iz]  - \
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
                         C5z * ((*d)[it][ix ][iz + 5 ] + (*d)[it][ix ][iz - 5 ]))*(*sp)[ix][iz];     
      }
    }
  }
}

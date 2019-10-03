#include <WaveReconV3.h>

WaveReconV3::WaveReconV3(const std::shared_ptr<SEP::float3DReg>model,
                         const std::shared_ptr<SEP::float3DReg>data,
                         const std::shared_ptr<SEP::float2DReg>slsqModel,
                         int                                    boundaryCond,
			 int					spongeWidth) {
std::cerr << "VERSION 3" << std::endl;
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

  // boundary condition must be 1 or 0
  assert(boundaryCond == 1 || boundaryCond == 0);
  _boundaryCond = boundaryCond;
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
  _dt2 = model->getHyper()->getAxis(3).d*model->getHyper()->getAxis(3).d;

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

  C0t = -2.0;
  C1t =  1.0;
  //C0t = -2.927222222;
  //C1t =  1.666666667;
  //C2t = -0.23809524;
  //C3t =  0.03968254;
  //C4t = -0.00496032;
  //C5t =  0.00031746;

  setDomainRange(model, data);

  buffer.reset(new SEP::float3DReg(data->getHyper()->getAxis(1).n+ 2 * _laplOrder,
                                    data->getHyper()->getAxis(2).n + 2 * _laplOrder,
                                    data->getHyper()->getAxis(3).n+2*_dt2Order));
  buffer->set(0);
  
  if(boundaryCond==1){
    _sponge.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n+2* _laplOrder,
                                      data->getHyper()->getAxis(2).n+2* _laplOrder));
  
    _sponge->set(1);
    int nz=data->getHyper()->getAxis(1).n;
    int nx=data->getHyper()->getAxis(2).n;
    int nt=data->getHyper()->getAxis(3).n;
  
    float alphaCos= 0.99;
    //for (int it = 0; it < nt; it++){
    for (int ix = _laplOrder; ix < nx+_laplOrder; ix++){
  		for (int iz = _laplOrder; iz < nz+_laplOrder; iz++) {
  			int distToEdge = std::min(std::min(ix,iz),std::min(nz+2*_laplOrder-iz-1,nx+2*_laplOrder-ix-1))-_laplOrder;
  			 if(distToEdge < _spongeWidth){
  		           float arg = M_PI / (1.0 * _spongeWidth) * 1.0 * (_spongeWidth-distToEdge);
  		           float coeff = alphaCos + (1.0-alphaCos) * cos(arg);
	                   (*_sponge->_mat)[ix][iz] = std::pow(coeff,10); 
  			  // (*_sponge->_mat)[ix][iz] = coeff; 
  			 }
  			//if(it<_tmin){
  			//	float coeff = exp(-(_lambda*_lambda)*(_tmin-it)*(_tmin-it));
  			//	(*_sponge->_mat)[it][ix][iz] *= coeff; 
  			//}
  		}
    }
   // }
   int iz=nz/2;
   for (int ix = 0; ix < nx+2*_laplOrder; ix++){
        std::cerr << "sponge[" << ix << "][" << iz << "]=" << (*_sponge->_mat)[ix][iz] << std::endl;
   }
   int ix=nx/2;
   for (int iz = 0; iz < nz+2*_laplOrder; iz++){
        std::cerr << "sponge[" << ix << "][" << iz << "]=" << (*_sponge->_mat)[ix][iz] << std::endl;
   }
  }
  // set slowness
  _slsq_dt2 = std::make_shared<float2DReg>(slsqModel->getHyper()->getAxis(1), slsqModel->getHyper()->getAxis(2));
  for (int ix = 0; ix < slsqModel->getHyper()->getAxis(2).n; ix++){
		for (int iz = 0; iz < slsqModel->getHyper()->getAxis(1).n; iz++) {
			(*_slsq_dt2->_mat)[ix][iz] = (*slsqModel->_mat)[ix][iz]/_dt2;
		}
	}
}

// WAm=d
// W[d^2/dt^2(model)*s^2 -Lapl(model)]=data
void WaveReconV3::forward(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const{
  if(_boundaryCond==0){
    forwardBound0(add,model,data);   
  }
  else if(_boundaryCond==1){
    forwardBound1(add,model,data);   
  }
}

void WaveReconV3::forwardBound0(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const {
  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);

  std::shared_ptr<float3D> b =
    ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);
    const std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
     std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);

      std::shared_ptr<float2D> s =
       ((std::dynamic_pointer_cast<float2DReg>(_slsq_dt2))->_mat);
    // load buffer
  #pragma omp parallel for collapse(3)
  for (int it = 0; it < n3; it++) {
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*b)[it + _dt2Order][ix + _laplOrder][iz+ _laplOrder] = (*m)[it][ix][iz];
      }
    }
  }

  #pragma omp parallel for collapse(2)
  for (int ix = 0; ix < n2; ix++) { //x
    for (int iz = 0; iz < n1; iz++) { //z
      (*d)[n3-2][ix][iz] += (*b)[0 + _dt2Order][ix + _laplOrder][iz+ _laplOrder];
      (*d)[n3-1][ix][iz] += (*b)[1 + _dt2Order][ix + _laplOrder][iz+ _laplOrder];
    }
  }
  #pragma omp parallel for collapse(3)
  for (int it = 1; it < n3-1; it++) { //time
    for (int ix = 0; ix < n2; ix++) { //x
      for (int iz = 0; iz < n1; iz++) { //z
        (*d)[it-1][ix][iz] +=     //second time deriv
       			          (C0t* (*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder]+ \
                                  C1t * ((*b)[it+_dt2Order-1][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 1][ix + _laplOrder][iz+ _laplOrder]))*(*s)[ix][iz]  - \
                                  //laplacian
                                  (C0x *(*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                                  C1x * ((*b)[it+_dt2Order][ix + 1 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                                  C2x * ((*b)[it+_dt2Order][ix + 2 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                                  C3x * ((*b)[it+_dt2Order][ix + 3 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                                  C4x * ((*b)[it+_dt2Order][ix + 4 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                                  C5x * ((*b)[it+_dt2Order][ix + 5 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                                  C0z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder]) + \
                                  C1z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 1 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                                  C2z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 2 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                                  C3z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 3 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                                  C4z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 4 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                                  C5z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 5 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 5 + _laplOrder]));

      }
    }
  }

}
void WaveReconV3::forwardBound1(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const {
  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);

  std::shared_ptr<float3D> b =
    ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);
    const std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
     std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
     std::shared_ptr<float2D> sp = _sponge->_mat;
      std::shared_ptr<float2D> s =
       ((std::dynamic_pointer_cast<float2DReg>(_slsq_dt2))->_mat);
    // load buffer
  #pragma omp parallel for collapse(3)
  for (int it = 0; it < n3; it++) {
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*b)[it + _dt2Order][ix + _laplOrder][iz+ _laplOrder] = (*m)[it][ix][iz];
      }
    }
  }

  #pragma omp parallel for collapse(2)
  for (int ix = 0; ix < n2; ix++) { //x
    for (int iz = 0; iz < n1; iz++) { //z
      (*d)[n3-2][ix][iz] += (*b)[0 + _dt2Order][ix + _laplOrder][iz+ _laplOrder];
      (*d)[n3-1][ix][iz] += (*b)[1 + _dt2Order][ix + _laplOrder][iz+ _laplOrder];
    }
  }
  #pragma omp parallel for collapse(3)
  for (int it = 1; it < n3-1; it++) { //time
    for (int ix = 0; ix < n2; ix++) { //x
      for (int iz = 0; iz < n1; iz++) { //z
        (*d)[it-1][ix][iz] +=     //second time deriv
       			          (C0t*(*sp)[ix+ _laplOrder][iz+ _laplOrder]* (*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder]+ \
                                  C1t *((*sp)[ix+ _laplOrder][iz+ _laplOrder]*(*b)[it+_dt2Order-1][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 1][ix + _laplOrder][iz+ _laplOrder]))*(*s)[ix][iz] - \
                                  //laplacian
                                  (C0x * (*sp)[ix+ _laplOrder][iz+ _laplOrder]*(*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                                  C1x * ((*sp)[ix+ _laplOrder+1][iz+ _laplOrder]*(*b)[it+_dt2Order][ix + 1 + _laplOrder][iz + _laplOrder] + (*sp)[ix+ _laplOrder-1][iz+ _laplOrder]*(*b)[it+_dt2Order][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                                  C2x * ((*sp)[ix+ _laplOrder+2][iz+ _laplOrder]*(*b)[it+_dt2Order][ix + 2 + _laplOrder][iz + _laplOrder] + (*sp)[ix+ _laplOrder-2][iz+ _laplOrder]*(*b)[it+_dt2Order][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                                  C3x * ((*sp)[ix+ _laplOrder+3][iz+ _laplOrder]*(*b)[it+_dt2Order][ix + 3 + _laplOrder][iz + _laplOrder] + (*sp)[ix+ _laplOrder-3][iz+ _laplOrder]*(*b)[it+_dt2Order][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                                  C4x * ((*sp)[ix+ _laplOrder+4][iz+ _laplOrder]*(*b)[it+_dt2Order][ix + 4 + _laplOrder][iz + _laplOrder] + (*sp)[ix+ _laplOrder-4][iz+ _laplOrder]*(*b)[it+_dt2Order][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                                  C5x * ((*sp)[ix+ _laplOrder+5][iz+ _laplOrder]*(*b)[it+_dt2Order][ix + 5 + _laplOrder][iz + _laplOrder] + (*sp)[ix+ _laplOrder-5][iz+ _laplOrder]*(*b)[it+_dt2Order][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                                  C0z * ((*sp)[ix+ _laplOrder][iz+ _laplOrder]*(*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder]) + \
                                  C1z * ((*sp)[ix+ _laplOrder][iz+ _laplOrder+1]*(*b)[it+_dt2Order][ix + _laplOrder][iz + 1 + _laplOrder] + (*sp)[ix+ _laplOrder][iz+ _laplOrder-1]*(*b)[it+_dt2Order][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                                  C2z * ((*sp)[ix+ _laplOrder][iz+ _laplOrder+2]*(*b)[it+_dt2Order][ix + _laplOrder][iz + 2 + _laplOrder] + (*sp)[ix+ _laplOrder][iz+ _laplOrder-2]*(*b)[it+_dt2Order][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                                  C3z * ((*sp)[ix+ _laplOrder][iz+ _laplOrder+3]*(*b)[it+_dt2Order][ix + _laplOrder][iz + 3 + _laplOrder] + (*sp)[ix+ _laplOrder][iz+ _laplOrder-3]*(*b)[it+_dt2Order][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                                  C4z * ((*sp)[ix+ _laplOrder][iz+ _laplOrder+4]*(*b)[it+_dt2Order][ix + _laplOrder][iz + 4 + _laplOrder] + (*sp)[ix+ _laplOrder][iz+ _laplOrder-4]*(*b)[it+_dt2Order][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                                  C5z * ((*sp)[ix+ _laplOrder][iz+ _laplOrder+5]*(*b)[it+_dt2Order][ix + _laplOrder][iz + 5 + _laplOrder] + (*sp)[ix+ _laplOrder][iz+ _laplOrder-5]*(*b)[it+_dt2Order][ix + _laplOrder][iz - 5 + _laplOrder]));

      }
    }
  }

}

// A*W*d=m
// (d^2/dt^2)*(W(data))*s^2 -Lapl*(W(data))]=model
void WaveReconV3::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const{
  if(_boundaryCond==0){
    adjointBound0(add,model,data);   
  }
  else if(_boundaryCond==1){
    adjointBound1(add,model,data);   
  }
}

void WaveReconV3::adjointBound0(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> b =
    ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);
    std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
    const std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
      std::shared_ptr<float2D> s =
       ((std::dynamic_pointer_cast<float2DReg>(_slsq_dt2))->_mat);

    // load buffer
    #pragma omp parallel for collapse(3)
    for (int it = 0; it < n3; it++) {
      for (int ix = 0; ix < n2; ix++) {
        for (int iz = 0; iz < n1; iz++) {
          (*b)[it + _dt2Order][ix + _laplOrder][iz+ _laplOrder] = (*d)[it][ix][iz];
        }
      }
    }

  #pragma omp parallel for collapse(2)
  for (int ix = 0; ix < n2; ix++) { //x
    for (int iz = 0; iz < n1; iz++) { //z
      (*m)[0][ix][iz] +=   (*b)[n3-2+ _dt2Order][ix + _laplOrder][iz+ _laplOrder]+C1t*(*b)[0+ _dt2Order][ix + _laplOrder][iz+ _laplOrder]*(*s)[ix][iz];
      (*m)[1][ix][iz] +=   (*b)[n3-1+ _dt2Order][ix + _laplOrder][iz+ _laplOrder]+(C0t*(*b)[0+ _dt2Order][ix + _laplOrder][iz+ _laplOrder]+C1t*(*b)[1+ _dt2Order][ix + _laplOrder][iz+ _laplOrder])*(*s)[ix][iz] -
                            //laplacian
                           (C0x * (*b)[0+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1x * ((*b)[0+_dt2Order][ix + 1 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                            C2x * ((*b)[0+_dt2Order][ix + 2 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                            C3x * ((*b)[0+_dt2Order][ix + 3 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                            C4x * ((*b)[0+_dt2Order][ix + 4 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                            C5x * ((*b)[0+_dt2Order][ix + 5 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                            C0z * (*b)[0+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 1 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                            C2z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 2 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                            C3z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 3 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                            C4z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 4 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                            C5z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 5 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 5 + _laplOrder]));
      (*m)[n3-2][ix][iz] += (C0t*(*b)[n3-3+ _dt2Order][ix + _laplOrder][iz+ _laplOrder] + C1t*(*b)[n3-4+ _dt2Order][ix + _laplOrder][iz+ _laplOrder])*(*s)[ix][iz] -
                            //laplacian
                           (C0x * (*b)[n3-3+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1x * ((*b)[n3-3+_dt2Order][ix + 1 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                            C2x * ((*b)[n3-3+_dt2Order][ix + 2 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                            C3x * ((*b)[n3-3+_dt2Order][ix + 3 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                            C4x * ((*b)[n3-3+_dt2Order][ix + 4 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                            C5x * ((*b)[n3-3+_dt2Order][ix + 5 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                            C0z * (*b)[n3-3+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 1 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                            C2z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 2 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                            C3z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 3 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                            C4z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 4 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                            C5z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 5 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 5 + _laplOrder]));
      (*m)[n3-1][ix][iz] += C1t*(*b)[n3-3+ _dt2Order][ix + _laplOrder][iz+ _laplOrder]*(*s)[ix][iz];
    }
  }

  #pragma omp parallel for collapse(3)
  for (int it = 1; it < n3-3; it++) {
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*m)[it+1][ix][iz] +=       //second time deriv
       			       (C0t*(*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                                 C1t * ((*b)[it+_dt2Order-1][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 1][ix + _laplOrder][iz+ _laplOrder])) *(*s)[ix][iz] - \
                            //laplacian
                            (C0x * (*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1x * ((*b)[it+_dt2Order][ix + 1 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                            C2x * ((*b)[it+_dt2Order][ix + 2 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                            C3x * ((*b)[it+_dt2Order][ix + 3 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                            C4x * ((*b)[it+_dt2Order][ix + 4 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                            C5x * ((*b)[it+_dt2Order][ix + 5 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                            C0z * (*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 1 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                            C2z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 2 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                            C3z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 3 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                            C4z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 4 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                            C5z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 5 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 5 + _laplOrder]));
      }
    }
  }
}
void WaveReconV3::adjointBound1(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> b =
    ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);
    std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
    const std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
      std::shared_ptr<float2D> s =
       ((std::dynamic_pointer_cast<float2DReg>(_slsq_dt2))->_mat);
     std::shared_ptr<float2D> sp = _sponge->_mat;

    // load buffer
    #pragma omp parallel for collapse(3)
    for (int it = 0; it < n3; it++) {
      for (int ix = 0; ix < n2; ix++) {
        for (int iz = 0; iz < n1; iz++) {
          (*b)[it + _dt2Order][ix + _laplOrder][iz+ _laplOrder] = (*d)[it][ix][iz];
        }
      }
    }

  #pragma omp parallel for collapse(2)
  for (int ix = 0; ix < n2; ix++) { //x
    for (int iz = 0; iz < n1; iz++) { //z
      (*m)[0][ix][iz] +=   (*b)[n3-2+ _dt2Order][ix + _laplOrder][iz+ _laplOrder]+ \
			 C1t*(*b)[0+ _dt2Order][ix + _laplOrder][iz+ _laplOrder]*(*s)[ix][iz]*(*sp)[ix+ _laplOrder][iz+ _laplOrder];
      (*m)[1][ix][iz] +=   (*b)[n3-1+ _dt2Order][ix + _laplOrder][iz+ _laplOrder] + \ 
			 (C0t*(*b)[0+ _dt2Order][ix + _laplOrder][iz+ _laplOrder]+C1t*(*b)[1+ _dt2Order][ix + _laplOrder][iz+ _laplOrder])*(*s)[ix][iz]*(*sp)[ix+ _laplOrder][iz+ _laplOrder] -
                            //laplacian
             (*sp)[ix+ _laplOrder][iz+ _laplOrder]*(C0x * (*b)[0+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1x * ((*b)[0+_dt2Order][ix + 1 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                            C2x * ((*b)[0+_dt2Order][ix + 2 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                            C3x * ((*b)[0+_dt2Order][ix + 3 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                            C4x * ((*b)[0+_dt2Order][ix + 4 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                            C5x * ((*b)[0+_dt2Order][ix + 5 + _laplOrder][iz + _laplOrder] + (*b)[0+_dt2Order][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                            C0z * (*b)[0+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 1 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                            C2z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 2 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                            C3z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 3 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                            C4z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 4 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                            C5z * ((*b)[0+_dt2Order][ix + _laplOrder][iz + 5 + _laplOrder] + (*b)[0+_dt2Order][ix + _laplOrder][iz - 5 + _laplOrder]));
      (*m)[n3-2][ix][iz] +=  (C0t*(*sp)[ix+ _laplOrder][iz+ _laplOrder]*(*b)[n3-3+ _dt2Order][ix + _laplOrder][iz+ _laplOrder] + C1t*(*b)[n3-4+ _dt2Order][ix + _laplOrder][iz+ _laplOrder])*(*s)[ix][iz] -
                            //laplacian
             (*sp)[ix+ _laplOrder][iz+ _laplOrder]*(C0x * (*b)[n3-3+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1x * ((*b)[n3-3+_dt2Order][ix + 1 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                            C2x * ((*b)[n3-3+_dt2Order][ix + 2 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                            C3x * ((*b)[n3-3+_dt2Order][ix + 3 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                            C4x * ((*b)[n3-3+_dt2Order][ix + 4 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                            C5x * ((*b)[n3-3+_dt2Order][ix + 5 + _laplOrder][iz + _laplOrder] + (*b)[n3-3+_dt2Order][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                            C0z * (*b)[n3-3+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 1 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                            C2z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 2 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                            C3z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 3 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                            C4z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 4 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                            C5z * ((*b)[n3-3+_dt2Order][ix + _laplOrder][iz + 5 + _laplOrder] + (*b)[n3-3+_dt2Order][ix + _laplOrder][iz - 5 + _laplOrder]));
      (*m)[n3-1][ix][iz] += C1t*(*b)[n3-3+ _dt2Order][ix + _laplOrder][iz+ _laplOrder]*(*s)[ix][iz];
    }
  }

  #pragma omp parallel for collapse(3)
  for (int it = 1; it < n3-3; it++) {
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*m)[it+1][ix][iz] +=       //second time deriv
       			       (C0t* (*sp)[ix+ _laplOrder][iz+ _laplOrder]*(*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                                 C1t * ((*b)[it+_dt2Order-1][ix + _laplOrder][iz + _laplOrder]+(*sp)[ix+ _laplOrder][iz+ _laplOrder]*(*b)[it+_dt2Order + 1][ix + _laplOrder][iz+ _laplOrder])) *(*s)[ix][iz] - \
                            //laplacian
             (*sp)[ix+ _laplOrder][iz+ _laplOrder]*(C0x * (*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1x * ((*b)[it+_dt2Order][ix + 1 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                            C2x * ((*b)[it+_dt2Order][ix + 2 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                            C3x * ((*b)[it+_dt2Order][ix + 3 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                            C4x * ((*b)[it+_dt2Order][ix + 4 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                            C5x * ((*b)[it+_dt2Order][ix + 5 + _laplOrder][iz + _laplOrder] + (*b)[it+_dt2Order][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                            C0z * (*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 1 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                            C2z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 2 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                            C3z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 3 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                            C4z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 4 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                            C5z * ((*b)[it+_dt2Order][ix + _laplOrder][iz + 5 + _laplOrder] + (*b)[it+_dt2Order][ix + _laplOrder][iz - 5 + _laplOrder]));
      }
    }
  }
}

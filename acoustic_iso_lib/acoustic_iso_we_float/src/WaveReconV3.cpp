#include <WaveReconV3.h>

WaveReconV3::WaveReconV3(const std::shared_ptr<SEP::float3DReg>model,
                         const std::shared_ptr<SEP::float3DReg>data,
                         const std::shared_ptr<SEP::float2DReg>slsqModel,
                         int                                    n1min,
                         int                                    n1max,
                         int                                    n2min,
                         int                                    n2max,
                         int                                    n3min,
                         int                                    n3max,
                         int                                    boundaryCond) {
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

  n1 =model->getHyper()->getAxis(1).n; //z
   n2 =model->getHyper()->getAxis(2).n; //x
   n3 =model->getHyper()->getAxis(3).n; //t

  // set domain and range
  setDomainRange(model, data);

  _n1min = n1min;
  _n1max = n1max;

  _n2min = n2min;
  _n2max = n2max;

  _n3min = n3min;
  _n3max = n3max;

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

  // C0t = -2.72222222;
  // C1t =  1.50000000;
  // C2t = -0.1500000;
  // C3t =  0.01111111;
  C0t = -2.927222222;
  C1t =  1.666666667;
  C2t = -0.23809524;
  C3t =  0.03968254;
  C4t = -0.00496032;
  C5t =  0.00031746;

  setDomainRange(model, data);

  buffer.reset(new SEP::float3DReg(data->getHyper()->getAxis(1).n+ 2 * _laplOrder,
                                    data->getHyper()->getAxis(2).n + 2 * _laplOrder,
                                    data->getHyper()->getAxis(3).n+2*_dt2Order));
  buffer->scale(0);

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

  // multiply by slowness squared
  #pragma omp parallel for collapse(3)

  for (int it = 0; it < n3; it++) { //time
    for (int ix = 0; ix < n2; ix++) { //x
      for (int iz = 0; iz < n1; iz++) { //z
        (*d)[it][ix][iz] +=       //second time deriv
        (C0t* (*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                                  C1t * ((*b)[it+_dt2Order-1][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 1][ix + _laplOrder][iz+ _laplOrder]) + \
                                  C2t * ((*b)[it+_dt2Order-2][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 2][ix + _laplOrder][iz+ _laplOrder]) + \
                                  C3t * ((*b)[it+_dt2Order-3][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 3][ix + _laplOrder][iz+ _laplOrder]) + \
                                  C4t * ((*b)[it+_dt2Order-4][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 4][ix + _laplOrder][iz+ _laplOrder]) + \
                                  C5t * ((*b)[it+_dt2Order-5][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 5][ix + _laplOrder][iz+ _laplOrder]))*(*s)[ix][iz] -
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

// A*W*d=m
// (d^2/dt^2)*(W(data))*s^2 -Lapl*(W(data))]=model
void WaveReconV3::adjoint(const bool                         add,
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

  // multiply by slowness squared
  #pragma omp parallel for collapse(3)

  for (int it = 0; it < n3; it++) {
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*m)[it][ix][iz] +=       //second time deriv
                            (C0t* (*b)[it+_dt2Order][ix + _laplOrder][iz + _laplOrder] + \
                            C1t * ((*b)[it+_dt2Order-1][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 1][ix + _laplOrder][iz+ _laplOrder]) + \
                            C2t * ((*b)[it+_dt2Order-2][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 2][ix + _laplOrder][iz+ _laplOrder]) + \
                            C3t * ((*b)[it+_dt2Order-3][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 3][ix + _laplOrder][iz+ _laplOrder]) + \
                            C4t * ((*b)[it+_dt2Order-4][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 4][ix + _laplOrder][iz+ _laplOrder]) + \
                            C5t * ((*b)[it+_dt2Order-5][ix + _laplOrder][iz + _laplOrder]+(*b)[it+_dt2Order + 5][ix + _laplOrder][iz+ _laplOrder]))*(*s)[ix][iz] -
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

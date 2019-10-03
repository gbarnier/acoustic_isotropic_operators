#include <Laplacian2d.h>


/**
   Many 2d slices case
 */
Laplacian2d::Laplacian2d(const std::shared_ptr<float3DReg>model,
                         const std::shared_ptr<float3DReg>data) {
  // ensure dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  n1 =model->getHyper()->getAxis(1).n; //z
   n2 =model->getHyper()->getAxis(2).n; //x
   n3 =model->getHyper()->getAxis(3).n; //t

  setDomainRange(model, data);

  // get spatial sampling
  _db = model->getHyper()->getAxis(1).d;
  _da = model->getHyper()->getAxis(2).d;

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

  // set domain and range
  setDomainRange(model, data);

  buffer.reset(new SEP::float3DReg(data->getHyper()->getAxis(1).n+ 2 * _laplOrder,
                                    data->getHyper()->getAxis(2).n + 2 * _laplOrder,
                                    data->getHyper()->getAxis(3).n));
  buffer->scale(0);
}

void Laplacian2d::forward(const bool                         add,
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

    // load buffer
  #pragma omp parallel for collapse(3)
  for (int it = 0; it < n3; it++) {
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*b)[it ][ix + _laplOrder][iz+ _laplOrder] = (*m)[it][ix][iz];
      }
    }
  }

  // multiply by slowness squared
  #pragma omp parallel for collapse(3)

  for (int it = 0; it < n3; it++) { //time
    for (int ix = 0; ix < n2; ix++) { //x
      for (int iz = 0; iz < n1; iz++) { //z
        (*d)[it][ix][iz] += //laplacian
                                  (C0x * (*b)[it][ix + _laplOrder][iz + _laplOrder] + \
                                  C1x * ((*b)[it][ix + 1 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                                  C2x * ((*b)[it][ix + 2 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                                  C3x * ((*b)[it][ix + 3 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                                  C4x * ((*b)[it][ix + 4 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                                  C5x * ((*b)[it][ix + 5 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                                  C0z * (*b)[it][ix + _laplOrder][iz + _laplOrder] + \
                                  C1z * ((*b)[it][ix + _laplOrder][iz + 1 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                                  C2z * ((*b)[it][ix + _laplOrder][iz + 2 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                                  C3z * ((*b)[it][ix + _laplOrder][iz + 3 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                                  C4z * ((*b)[it][ix + _laplOrder][iz + 4 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                                  C5z * ((*b)[it][ix + _laplOrder][iz + 5 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 5 + _laplOrder]));
      }
    }
  }

}

void Laplacian2d::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const {
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> b =
    ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);
    std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
    const std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);

    // load buffer
    #pragma omp parallel for collapse(3)
    for (int it = 0; it < n3; it++) {
      for (int ix = 0; ix < n2; ix++) {
        for (int iz = 0; iz < n1; iz++) {
          (*b)[it ][ix + _laplOrder][iz+ _laplOrder] = (*d)[it][ix][iz];
        }
      }
    }

  // multiply by slowness squared
  #pragma omp parallel for collapse(3)

  for (int it = 0; it < n3; it++) {
    for (int ix = 0; ix < n2; ix++) {
      for (int iz = 0; iz < n1; iz++) {
        (*m)[it][ix][iz] +=       //laplacian
                            (C0x * (*b)[it][ix + _laplOrder][iz + _laplOrder] + \
                            C1x * ((*b)[it][ix + 1 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 1 + _laplOrder][iz + _laplOrder]) + \
                            C2x * ((*b)[it][ix + 2 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 2 + _laplOrder][iz + _laplOrder]) + \
                            C3x * ((*b)[it][ix + 3 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 3 + _laplOrder][iz + _laplOrder]) + \
                            C4x * ((*b)[it][ix + 4 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 4 + _laplOrder][iz + _laplOrder]) + \
                            C5x * ((*b)[it][ix + 5 + _laplOrder][iz + _laplOrder] + (*b)[it][ix - 5 + _laplOrder][iz + _laplOrder]) + \
                            C0z * (*b)[it][ix + _laplOrder][iz + _laplOrder] + \
                            C1z * ((*b)[it][ix + _laplOrder][iz + 1 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 1 + _laplOrder]) + \
                            C2z * ((*b)[it][ix + _laplOrder][iz + 2 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 2 + _laplOrder]) + \
                            C3z * ((*b)[it][ix + _laplOrder][iz + 3 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 3 + _laplOrder]) + \
                            C4z * ((*b)[it][ix + _laplOrder][iz + 4 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 4 + _laplOrder]) + \
                            C5z * ((*b)[it][ix + _laplOrder][iz + 5 + _laplOrder] + (*b)[it][ix + _laplOrder][iz - 5 + _laplOrder]));
      }
    }
  }
}

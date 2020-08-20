#include <SecondDeriv.h>
using namespace SEP;

// /**
//    2d case
//  */
// SecondDeriv::SecondDeriv(const std::shared_ptr<float2DReg>model,
//                          const std::shared_ptr<float2DReg>data) {
//   // ensure dimensions match
//   assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
//   assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
//   assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
//   assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
//
//   // set dim
//   _dim = 2;
//
//   // get sampling of fast axis and square it
//   _df2 = model->getHyper()->getAxis(1).d * model->getHyper()->getAxis(1).d;
//
//   // set domain and range
//   setDomainRange(model, data);
// }

/**
   Many 2d slices case
 */
SecondDeriv::SecondDeriv(const std::shared_ptr<float3DReg>model,
                         const std::shared_ptr<float3DReg>data) {
  // ensure dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  // set dim
  _dim = 3;

  // get sampling of fast axis and square it
  _df2 = model->getHyper()->getAxis(1).d * model->getHyper()->getAxis(1).d;

  setDomainRange(model, data);
}

void SecondDeriv::forward(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const{
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  float *modelVals, *dataVals;
  int    n1, n2, n3;

  if (_dim == 3) {
    modelVals =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat)->data();
    dataVals =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat)->data();
    n1 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
    n2 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
    n3 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;
  }
  else if (_dim == 2) {
    modelVals =
      ((std::dynamic_pointer_cast<float2DReg>(model))->_mat)->data();
    dataVals =
      ((std::dynamic_pointer_cast<float2DReg>(data))->_mat)->data();
    n1 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
    n2 =
      (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;
    n3 = 1;
  }
#pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      dataVals[i2 * n1 + i3 * n2 * n1] += (-2 *
                                           modelVals[i2 * n1 + i3 * n2 *
                                                     n1] +
                                           modelVals[1 + i2 * n1 + i3 * n2 *
                                                     n1]) / _df2;
      dataVals[n1 - 1 + i2 * n1 + i3 * n2 * n1] += (-2 *
                                                    modelVals[n1 - 1 + i2 *
                                                              n1 + i3 * n2 *
                                                              n1] +
                                                    modelVals[n1 - 2 + i2 *
                                                              n1 + i3 * n2 *
                                                              n1]) / _df2;
#pragma omp parallel for

      for (int i1 = 1; i1 < n1 - 1; i1++) {
        dataVals[i1 + i2 * n1 + i3 * n2 *
                 n1] += (modelVals[i1 - 1 + i2 * n1 + i3 * n2 * n1] +
                         -2 * modelVals[i1 + i2 * n1 + i3 * n2 * n1] +
                         modelVals[i1 + 1 + i2 * n1 + i3 * n2 * n1]) / _df2;
      }
    }
  }
}

void SecondDeriv::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  float *modelVals, *dataVals;
  int    n1, n2, n3;

  if (_dim == 3) {
    modelVals =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat)->data();
    dataVals =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat)->data();
    n1 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
    n2 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
    n3 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;
  }
  else if (_dim == 2) {
    modelVals =
      ((std::dynamic_pointer_cast<float2DReg>(model))->_mat)->data();
    dataVals =
      ((std::dynamic_pointer_cast<float2DReg>(data))->_mat)->data();
    n1 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
    n2 =
      (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;
    n3 = 1;
  }
#pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      modelVals[i2 * n1 + i3 * n2 * n1] += (-2 *
                                            dataVals[i2 * n1 + i3 * n2 *
                                                     n1] +
                                            dataVals[1 + i2 * n1 + i3 * n2 *
                                                     n1]) / _df2;
      modelVals[n1 - 1 + i2 * n1 + i3 * n2 * n1] += (-2 *
                                                     dataVals[n1 - 1 + i2 *
                                                              n1 + i3 * n2 *
                                                              n1] +
                                                     dataVals[n1 - 2 + i2 *
                                                              n1 + i3 * n2 *
                                                              n1]) / _df2;
#pragma omp parallel for

      for (int i1 = 1; i1 < n1 - 1; i1++) {
        modelVals[i1 + i2 * n1 + i3 * n2 *
                  n1] += (dataVals[i1 - 1 + i2 * n1 + i3 * n2 * n1] +
                          -2 *
                          dataVals[i1 + i2 * n1 + i3 * n2 * n1] +
                          dataVals[i1 + 1 + i2 *
                                   n1 + i3 * n2 * n1]) / _df2;
      }
    }
  }
}

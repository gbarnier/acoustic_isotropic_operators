#include <Laplacian2d.h>


/**
   One 2d slice case
 */
// Laplacian2d::Laplacian2d(const std::shared_ptr<float2DReg>model,
//                          const std::shared_ptr<float2DReg>data) {
//   // ensure dimensions match
//   assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
//   assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
//   assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
//   assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
//
//   // get spatial sampling
//   _db = model->getHyper()->getAxis(2).d;
//   _da = model->getHyper()->getAxis(1).d;
//
//   // set 3d flag
//   _3d = false;
//
//   // set domain and range
//   setDomainRange(model, data);
//
//   // calculate lapl coefficients
//   C0z = -2.927222222 / (_db * _db);
//   C1z =  1.666666667 / (_db * _db);
//   C2z = -0.238095238 / (_db * _db);
//   C3z =  0.039682539 / (_db * _db);
//   C4z = -0.004960317 / (_db * _db);
//   C5z =  0.000317460 / (_db * _db);
//
//   C0x = -2.927222222 / (_da * _da);
//   C1x =  1.666666667 / (_da * _da);
//   C2x = -0.238095238 / (_da * _da);
//   C3x =  0.039682539 / (_da * _da);
//   C4x = -0.004960317 / (_da * _da);
//   C5x =  0.000317460 / (_da * _da);
//
//   buffer.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n + 2 *
//                                     _bufferSize,
//                                     data->getHyper()->getAxis(2).n + 2 *
//                                     _bufferSize));
//   buffer->scale(0);
// }

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

  // get spatial sampling
  _db = model->getHyper()->getAxis(3).d;
  _da = model->getHyper()->getAxis(2).d;

  // set 3d flag
  _3d = true;

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

  setDomainRange(model, data);

  buffer.reset(new SEP::float3DReg(data->getHyper()->getAxis(1).n,
                                    data->getHyper()->getAxis(2).n + 2 *
                                    _bufferSize,
                                    data->getHyper()->getAxis(3).n + 2 *
                                    _bufferSize));
  buffer->scale(0);
}

void Laplacian2d::forward(const bool                         add,
                          const std::shared_ptr<SEP::float3DReg>model,
                          std::shared_ptr<SEP::float3DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  int i1, i2, i3;

  if (_3d) {
    const std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
    std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
    int n1 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
    int n2 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
    int n3 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;

    // std::shared_ptr<float3DReg> buffer(new  float3DReg(n1,
    //                                                    n2 + 2 * _bufferSize,
    //                                                    n3 + 2 *
    // _bufferSize));
    //
    // buffer->scale(0);
    std::shared_ptr<float3D> b =
      ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);

    #pragma omp parallel for collapse(3)

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          (*b)[i3 + _bufferSize][i2 + _bufferSize][i1] = (*m)[i3][i2][i1];
        }
      }
    }


    // calculate laplacian. This will be the data entry in the gradio operator
    #pragma omp parallel for collapse(3)

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          (*d)[i3][i2][i1] +=       C0z *
                              (*b)[i3 + _bufferSize][i2 + _bufferSize][i1] + \
                              C1z *
                              ((*b)[i3 + 1 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 1 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C2z *
                              ((*b)[i3 + 2 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 2 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C3z *
                              ((*b)[i3 + 3 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 3 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C4z *
                              ((*b)[i3 + 4 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 4 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C5z *
                              ((*b)[i3 + 5 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 5 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C0x *
                              (*b)[i3 + _bufferSize][i2 + _bufferSize][i1] + \
                              C1x *
                              ((*b)[i3 + _bufferSize][i2 + 1 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 1 + _bufferSize][i1]) + \
                              C2x *
                              ((*b)[i3 + _bufferSize][i2 + 2 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 2 + _bufferSize][i1]) + \
                              C3x *
                              ((*b)[i3 + _bufferSize][i2 + 3 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 3 + _bufferSize][i1]) + \
                              C4x *
                              ((*b)[i3 + _bufferSize][i2 + 4 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 4 + _bufferSize][i1]) + \
                              C5x *
                              ((*b)[i3 + _bufferSize][i2 + 5 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 5 + _bufferSize][i1]);
        }
      }
    }
  }
  else {
    const std::shared_ptr<float2D> m =
      ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
    std::shared_ptr<float2D> d =
      ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
    int n2 =
      (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
    int n3 =
      (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;

    // std::shared_ptr<float2DReg> buffer(new  float2DReg(n2 + 2 * _bufferSize,
    //                                                    n3 + 2 *
    // _bufferSize));
    //
    // buffer->scale(0);
    std::shared_ptr<float2D> b =
      ((std::dynamic_pointer_cast<float2DReg>(buffer))->_mat);

    #pragma omp parallel for collapse(2)

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        (*b)[i3 + _bufferSize][i2 + _bufferSize] = (*m)[i3][i2];
      }
    }


    // calculate laplacian. This will be the data entry in the gradio operator
    #pragma omp parallel for collapse(2)

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        (*d)[i3][i2] +=       C0z *  (*b)[i3 + _bufferSize][i2 + _bufferSize] + \
                        C1z *
                        ((*b)[i3 + 1 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 1 + _bufferSize][i2 + _bufferSize]) + \
                        C2z *
                        ((*b)[i3 + 2 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 2 + _bufferSize][i2 + _bufferSize]) + \
                        C3z *
                        ((*b)[i3 + 3 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 3 + _bufferSize][i2 + _bufferSize]) + \
                        C4z *
                        ((*b)[i3 + 4 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 4 + _bufferSize][i2 + _bufferSize]) + \
                        C5z *
                        ((*b)[i3 + 5 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 5 + _bufferSize][i2 + _bufferSize]) +  \
                        C0x *  (*b)[i3 + _bufferSize][i2 + _bufferSize] + \
                        C1x *
                        ((*b)[i3 + _bufferSize][i2 + 1 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 1 + _bufferSize]) + \
                        C2x *
                        ((*b)[i3 + _bufferSize][i2 + 2 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 2 + _bufferSize]) + \
                        C3x *
                        ((*b)[i3 + _bufferSize][i2 + 3 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 3 + _bufferSize]) + \
                        C4x *
                        ((*b)[i3 + _bufferSize][i2 + 4 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 4 + _bufferSize]) + \
                        C5x *
                        ((*b)[i3 + _bufferSize][i2 + 5 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 5 + _bufferSize]);
      }
    }
  }
}

void Laplacian2d::adjoint(const bool                         add,
                          std::shared_ptr<SEP::float3DReg>      model,
                          const std::shared_ptr<SEP::float3DReg>data) const {
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  int i1, i2, i3;

  if (_3d) {
    std::shared_ptr<float3D> m =
      ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
    const std::shared_ptr<float3D> d =
      ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
    int n1 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
    int n2 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
    int n3 =
      (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;

    // std::shared_ptr<float3DReg> buffer(new  float3DReg(n1,
    //                                                    n2 + 2 * _bufferSize,
    //                                                    n3 + 2 *
    // _bufferSize));
    //
    // buffer->scale(0);
    std::shared_ptr<float3D> b =
      ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);


    #pragma omp parallel for collapse(3)

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          (*b)[i3 + _bufferSize][i2 + _bufferSize][i1] = (*d)[i3][i2][i1];
        }
      }
    }


    // calculate laplacian.
    #pragma omp parallel for collapse(3)

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          (*m)[i3][i2][i1] +=       C0z *
                              (*b)[i3 + _bufferSize][i2 + _bufferSize][i1] + \
                              C1z *
                              ((*b)[i3 + 1 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 1 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C2z *
                              ((*b)[i3 + 2 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 2 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C3z *
                              ((*b)[i3 + 3 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 3 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C4z *
                              ((*b)[i3 + 4 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 4 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C5z *
                              ((*b)[i3 + 5 + _bufferSize][i2 + _bufferSize][i1] +
                               (*b)[i3 - 5 + _bufferSize][i2 + _bufferSize][i1]) + \
                              C0x *
                              (*b)[i3 + _bufferSize][i2 + _bufferSize][i1] + \
                              C1x *
                              ((*b)[i3 + _bufferSize][i2 + 1 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 1 + _bufferSize][i1]) + \
                              C2x *
                              ((*b)[i3 + _bufferSize][i2 + 2 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 2 + _bufferSize][i1]) + \
                              C3x *
                              ((*b)[i3 + _bufferSize][i2 + 3 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 3 + _bufferSize][i1]) + \
                              C4x *
                              ((*b)[i3 + _bufferSize][i2 + 4 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 4 + _bufferSize][i1]) + \
                              C5x *
                              ((*b)[i3 + _bufferSize][i2 + 5 + _bufferSize][i1] +
                               (*b)[i3 + _bufferSize][i2 - 5 + _bufferSize][i1]);
        }
      }
    }
  }
  else {
    std::shared_ptr<float2D> m =
      ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
    const std::shared_ptr<float2D> d =
      ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
    int n2 =
      (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
    int n3 =
      (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;

    // std::shared_ptr<float2DReg> buffer(new  float2DReg(n2 + 2 * _bufferSize,
    //                                                    n3 + 2 *
    // _bufferSize));
    //
    // buffer->scale(0);
    std::shared_ptr<float2D> b =
      ((std::dynamic_pointer_cast<float2DReg>(buffer))->_mat);

    #pragma omp parallel for collapse(2)

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        (*b)[i3 + _bufferSize][i2 + _bufferSize] = (*d)[i3][i2];
      }
    }


    // calculate laplacian.
    #pragma omp parallel for collapse(2)

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        (*m)[i3][i2] +=       C0z *  (*b)[i3 + _bufferSize][i2 + _bufferSize] + \
                        C1z *
                        ((*b)[i3 + 1 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 1 + _bufferSize][i2 + _bufferSize]) + \
                        C2z *
                        ((*b)[i3 + 2 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 2 + _bufferSize][i2 + _bufferSize]) + \
                        C3z *
                        ((*b)[i3 + 3 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 3 + _bufferSize][i2 + _bufferSize]) + \
                        C4z *
                        ((*b)[i3 + 4 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 4 + _bufferSize][i2 + _bufferSize]) + \
                        C5z *
                        ((*b)[i3 + 5 + _bufferSize][i2 + _bufferSize] +
                         (*b)[i3 - 5 + _bufferSize][i2 + _bufferSize]) +  \
                        C0x *  (*b)[i3 + _bufferSize][i2 + _bufferSize] + \
                        C1x *
                        ((*b)[i3 + _bufferSize][i2 + 1 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 1 + _bufferSize]) + \
                        C2x *
                        ((*b)[i3 + _bufferSize][i2 + 2 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 2 + _bufferSize]) + \
                        C3x *
                        ((*b)[i3 + _bufferSize][i2 + 3 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 3 + _bufferSize]) + \
                        C4x *
                        ((*b)[i3 + _bufferSize][i2 + 4 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 4 + _bufferSize]) + \
                        C5x *
                        ((*b)[i3 + _bufferSize][i2 + 5 + _bufferSize] +
                         (*b)[i3 + _bufferSize][i2 - 5 + _bufferSize]);
      }
    }
  }
}

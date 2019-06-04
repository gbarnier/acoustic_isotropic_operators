#include <Laplacian2dFast.h>
using namespace waveform;
using namespace giee;

/**
   One 2d slice case
 */
Laplacian2dFast::Laplacian2dFast(const std::shared_ptr<float2DReg>model,
                                 const std::shared_ptr<float2DReg>data) {
  // ensure dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);

  // get spatial sampling
  _da = model->getHyper()->getAxis(2).d;
  _db = model->getHyper()->getAxis(1).d;

  // set 3d flag
  _3d = false;

  // set domain and range
  setDomainRange(model, data);

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
}

/**
   Many 2d slices case
 */
Laplacian2dFast::Laplacian2dFast(const std::shared_ptr<float3DReg>model,
                                 const std::shared_ptr<float3DReg>data) {
  // ensure dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  // get spatial sampling
  _da = model->getHyper()->getAxis(3).d;
  _db = model->getHyper()->getAxis(2).d;

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
}

void Laplacian2dFast::forward(const bool                         add,
                              const std::shared_ptr<giee::Vector>model,
                              std::shared_ptr<giee::Vector>      data) {
  assert(checkDomainRange(model, data, true));

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
    // buffer->scale(0);
    // std::shared_ptr<float3D> b =
    //   ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);

    // #pragma omp parallel for collapse(3)
    //
    // for (i3 = 0; i3 < n3; i3++) {
    //   for (i2 = 0; i2 < n2; i2++) {
    //     for (i1 = 0; i1 < n1; i1++) {
    //       (*b)[i3 + _bufferSize][i2 + _bufferSize][i1] = (*m)[i3][i2][i1];
    //     }
    //   }
    // }

    for (i3 = 0; i3 < _bufferSize; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          // (*d)[i3][i2][i1] += C0z * (*b)[i3][i2][i1] + C0x *
          // (*b)[i3][i2][i1];
          if (i3 == 0) {
            (*d)[0][i2][i1] += C0x * (*m)[0][i2][i1] +   \
                               C1x * ((*m)[1][i2][i1]) + \
                               C2x * ((*m)[2][i2][i1]) + \
                               C3x * ((*m)[3][i2][i1]) + \
                               C4x * ((*m)[4][i2][i1]) + \
                               C5x *   ((*m)[5][i2][i1]);
            (*d)[n3 - 1][i2][i1] += C0x * (*m)[n3 - 1][i2][i1] +   \
                                    C1x * ((*m)[n3 - 2][i2][i1]) + \
                                    C2x * ((*m)[n3 - 3][i2][i1]) + \
                                    C3x * ((*m)[n3 - 4][i2][i1]) + \
                                    C4x * ((*m)[n3 - 5][i2][i1]) + \
                                    C5x * ((*m)[n3 - 6][i2][i1]);
          }
          else if (i3 == 1) {
            (*d)[1][i2][i1] += C0x * (*m)[1][i2][i1] +                     \
                               C1x * ((*m)[2][i2][i1] + (*m)[0][i2][i1]) + \
                               C2x * ((*m)[3][i2][i1]) +                   \
                               C3x * ((*m)[4][i2][i1]) +                   \
                               C4x * ((*m)[5][i2][i1]) +                   \
                               C5x *   ((*m)[6][i2][i1]);
            (*d)[n3 - 2][i2][i1] += C0x * (*m)[n3 - 2][i2][i1] + \
                                    C1x *
                                    ((*m)[n3 - 3][i2][i1] +
                                     (*m)[n3 - 1][i2][i1]) +       \
                                    C2x * ((*m)[n3 - 4][i2][i1]) + \
                                    C3x * ((*m)[n3 - 5][i2][i1]) + \
                                    C4x * ((*m)[n3 - 6][i2][i1]) + \
                                    C5x * ((*m)[n3 - 7][i2][i1]);
          }
          else if (i3 == 2) {
            (*d)[2][i2][i1] += C0x * (*m)[2][i2][i1] + \
                               C1x *
                               ((*m)[1][i2][i1] + (*m)[3][i2][i1]) + \
                               C2x *
                               ((*m)[0][i2][i1] + (*m)[4][i2][i1]) + \
                               C3x * ((*m)[5][i2][i1]) +             \
                               C4x * ((*m)[6][i2][i1]) +             \
                               C5x *   ((*m)[7][i2][i1]);
            (*d)[n3 - 3][i2][i1] += C0x * (*m)[n3 - 3][i2][i1] + \
                                    C1x *
                                    ((*m)[n3 - 4][i2][i1] +
                                     (*m)[n3 - 2][i2][i1]) + \
                                    C2x *
                                    ((*m)[n3 - 5][i2][i1] +
                                     (*m)[n3 - 1][i2][i1]) +       \
                                    C3x * ((*m)[n3 - 6][i2][i1]) + \
                                    C4x * ((*m)[n3 - 7][i2][i1]) + \
                                    C5x * ((*m)[n3 - 8][i2][i1]);
          }
          else if (i3 == 3) {
            (*d)[3][i2][i1] += C0x * (*m)[i3][i2][i1] + \
                               C1x *
                               ((*m)[2][i2][i1] + (*m)[4][i2][i1]) + \
                               C2x *
                               ((*m)[1][i2][i1] + (*m)[5][i2][i1]) + \
                               C3x *
                               ((*m)[0][i2][i1] + (*m)[6][i2][i1]) + \
                               C4x * ((*m)[7][i2][i1]) +             \
                               C5x * ((*m)[8][i2][i1]);
            (*d)[n3 - 4][i2][i1] += C0x * (*m)[n3 - 4][i2][i1] + \
                                    C1x *
                                    ((*m)[n3 - 5][i2][i1] +
                                     (*m)[n3 - 3][i2][i1]) + \
                                    C2x *
                                    ((*m)[n3 - 6][i2][i1] +
                                     (*m)[n3 - 2][i2][i1]) + \
                                    C3x *
                                    ((*m)[n3 - 7][i2][i1] +
                                     (*m)[n3 - 1][i2][i1]) +       \
                                    C4x * ((*m)[n3 - 8][i2][i1]) + \
                                    C5x * ((*m)[n3 - 9][i2][i1]);
          }
          else if (i3 == 4) {
            (*d)[4][i2][i1] += C0x * (*m)[4][i2][i1] + \
                               C1x *
                               ((*m)[3][i2][i1] + (*m)[5][i2][i1]) + \
                               C2x *
                               ((*m)[2][i2][i1] + (*m)[6][i2][i1]) + \
                               C3x *
                               ((*m)[1][i2][i1] + (*m)[7][i2][i1]) + \
                               C4x *
                               ((*m)[0][i2][i1] + (*m)[8][i2][i1]) + \
                               C5x *   ((*m)[9][i2][i1]);
            (*d)[n3 - 5][i2][i1] += C0x * (*m)[n3 - 5][i2][i1] + \
                                    C1x *
                                    ((*m)[n3 - 6][i2][i1] +
                                     (*m)[n3 - 4][i2][i1]) + \
                                    C2x *
                                    ((*m)[n3 - 7][i2][i1] +
                                     (*m)[n3 - 3][i2][i1]) + \
                                    C3x *
                                    ((*m)[n3 - 8][i2][i1] +
                                     (*m)[n3 - 2][i2][i1]) + \
                                    C4x *
                                    ((*m)[n3 - 9][i2][i1] +
                                     (*m)[n3 - 1][i2][i1]) + \
                                    C5x *   ((*m)[n3 - 10][i2][i1]);
          }
        }
      }
    }

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < _bufferSize; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          if (i2 == 0) {
            (*d)[i3][0][i1] += C0z * (*m)[i3][0][i1] +   \
                               C1z * ((*m)[i3][1][i1]) + \
                               C2z * ((*m)[i3][2][i1]) + \
                               C3z * ((*m)[i3][3][i1]) + \
                               C4z * ((*m)[i3][4][i1]) + \
                               C5z * ((*m)[i3][5][i1]);
            (*d)[i3][n2 - 1][i1] += C0z * (*m)[i3][n2 - 1][i1] +   \
                                    C1z * ((*m)[i3][n2 - 2][i1]) + \
                                    C2z * ((*m)[i3][n2 - 3][i1]) + \
                                    C3z * ((*m)[i3][n2 - 4][i1]) + \
                                    C4z * ((*m)[i3][n2 - 5][i1]) + \
                                    C5z * ((*m)[i3][n2 - 6][i1]);
          }
          else if (i2 == 1) {
            (*d)[i3][1][i1] += C0z * (*m)[i3][i2][i1] + \
                               C1z *
                               ((*m)[i3][i2 + 1][i1] + (*m)[i3][i2 - 1][i1]) + \
                               C2z * ((*m)[i3][i2 + 2][i1]) +                  \
                               C3z * ((*m)[i3][i2 + 3][i1]) +                  \
                               C4z * ((*m)[i3][i2 + 4][i1]) +                  \
                               C5z * ((*m)[i3][i2 + 5][i1]);
            (*d)[i3][n2 - 2][i1] += C0z * (*m)[i3][n2 - 2][i1] + \
                                    C1z *
                                    ((*m)[i3][n2 - 3][i1] +
                                     (*m)[i3][n2 - 1][i1]) +       \
                                    C2z * ((*m)[i3][n2 - 4][i1]) + \
                                    C3z * ((*m)[i3][n2 - 5][i1]) + \
                                    C4z * ((*m)[i3][n2 - 6][i1]) + \
                                    C5z * ((*m)[i3][n2 - 7][i1]);
          }
          else if (i2 == 2) {
            (*d)[i3][2][i1] += C0z * (*m)[i3][2][i1] + \
                               C1z *
                               ((*m)[i3][1][i1] + (*m)[i3][3][i1]) + \
                               C2z *
                               ((*m)[i3][0][i1] + (*m)[i3][4][i1]) + \
                               C3z * ((*m)[i3][5][i1]) +             \
                               C4z * ((*m)[i3][6][i1]) +             \
                               C5z * ((*m)[i3][7][i1]);
            (*d)[i3][n2 - 3][i1] += C0z * (*m)[i3][n2 - 3][i1] + \
                                    C1z *
                                    ((*m)[i3][n2 - 4][i1] +
                                     (*m)[i3][n2 - 2][i1]) + \
                                    C2z *
                                    ((*m)[i3][n2 - 5][i1] +
                                     (*m)[i3][n2 - 1][i1]) +       \
                                    C3z * ((*m)[i3][n2 - 6][i1]) + \
                                    C4z * ((*m)[i3][n2 - 7][i1]) + \
                                    C5z * ((*m)[i3][n2 - 8][i1]);
          }
          else if (i2 == 3) {
            (*d)[i3][3][i1] += C0z * (*m)[i3][3][i1] + \
                               C1z *
                               ((*m)[i3][2][i1] + (*m)[i3][4][i1]) + \
                               C2z *
                               ((*m)[i3][1][i1] + (*m)[i3][5][i1]) + \
                               C3z *
                               ((*m)[i3][0][i1] + (*m)[i3][6][i1]) + \
                               C4z * ((*m)[i3][7][i1]) +             \
                               C5z * ((*m)[i3][8][i1]);
            (*d)[i3][n2 - 4][i1] += C0z * (*m)[i3][n2 - 4][i1] + \
                                    C1z *
                                    ((*m)[i3][n2 - 5][i1] +
                                     (*m)[i3][n2 - 3][i1]) + \
                                    C2z *
                                    ((*m)[i3][n2 - 6][i1] +
                                     (*m)[i3][n2 - 2][i1]) + \
                                    C3z *
                                    ((*m)[i3][n2 - 7][i1] +
                                     (*m)[i3][n2 - 1][i1]) +       \
                                    C4z * ((*m)[i3][n2 - 8][i1]) + \
                                    C5z * ((*m)[i3][n2 - 9][i1]);
          }
          else if (i2 == 4) {
            (*d)[i3][4][i1] += C0z * (*m)[i3][4][i1] + \
                               C1z *
                               ((*m)[i3][3][i1] + (*m)[i3][5][i1]) + \
                               C2z *
                               ((*m)[i3][2][i1] + (*m)[i3][6][i1]) + \
                               C3z *
                               ((*m)[i3][1][i1] + (*m)[i3][7][i1]) + \
                               C4z *
                               ((*m)[i3][0][i1] + (*m)[i3][8][i1]) + \
                               C5z *   ((*m)[i3][9][i1]);
            (*d)[i3][n2 - 5][i1] += C0z * (*m)[i3][n2 - 5][i1] + \
                                    C1z *
                                    ((*m)[i3][n2 - 6][i1] +
                                     (*m)[i3][n2 - 4][i1]) + \
                                    C2z *
                                    ((*m)[i3][n2 - 7][i1] +
                                     (*m)[i3][n2 - 3][i1]) + \
                                    C3z *
                                    ((*m)[i3][n2 - 8][i1] +
                                     (*m)[i3][n2 - 2][i1]) + \
                                    C4z *
                                    ((*m)[i3][n2 - 9][i1] +
                                     (*m)[i3][n2 - 1][i1]) + \
                                    C5z *   ((*m)[i3][n2 - 10][i1]);
          }
        }
      }
    }

    // calculate laplacian.
    #pragma omp parallel for collapse(3)

    for (i3 = _bufferSize; i3 < n3 - _bufferSize; i3++) {
      for (i2 = _bufferSize; i2 < n2 - _bufferSize; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          (*d)[i3][i2][i1] +=       C0x *
                              (*m)[i3][i2][i1] + \
                              C1x *
                              ((*m)[i3 + 1][i2][i1] +
                               (*m)[i3 - 1][i2][i1]) + \
                              C2x *
                              ((*m)[i3 + 2][i2][i1] +
                               (*m)[i3 - 2][i2][i1]) + \
                              C3x *
                              ((*m)[i3 + 3][i2][i1] +
                               (*m)[i3 - 3][i2][i1]) + \
                              C4x *
                              ((*m)[i3 + 4][i2][i1] +
                               (*m)[i3 - 4][i2][i1]) + \
                              C5x *
                              ((*m)[i3 + 5][i2][i1] +
                               (*m)[i3 - 5][i2][i1]) + \
                              C0z *
                              (*m)[i3][i2][i1] + \
                              C1z *
                              ((*m)[i3][i2 + 1][i1] +
                               (*m)[i3][i2 - 1][i1]) + \
                              C2z *
                              ((*m)[i3][i2 + 2][i1] +
                               (*m)[i3][i2 - 2][i1]) + \
                              C3z *
                              ((*m)[i3][i2 + 3][i1] +
                               (*m)[i3][i2 - 3][i1]) + \
                              C4z *
                              ((*m)[i3][i2 + 4][i1] +
                               (*m)[i3][i2 - 4][i1]) + \
                              C5z *
                              ((*m)[i3][i2 + 5][i1] +
                               (*m)[i3][i2 - 5][i1]);
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
    // buffer->scale(0);
    // std::shared_ptr<float2D> b =
    //   ((std::dynamic_pointer_cast<float2DReg>(buffer))->_mat);
    //
    // #pragma omp parallel for collapse(2)
    //
    // for (i3 = 0; i3 < n3; i3++) {
    //   for (i2 = 0; i2 < n2; i2++) {
    //     (*b)[i3 + _bufferSize][i2 + _bufferSize] = (*m)[i3][i2];
    //   }
    // }

    // for (i3 = 0; i3 < _bufferSize; i3++) {
    //   for (i2 = 0; i2 < n2; i2++) {
    //     // (*d)[i3][i2]  += C0z * (*b)[i3][i2]  + C0x *
    //     // (*b)[i3][i2] ;
    //     if (i3 == 0) {
    //       (*d)[0][i2] += C0x * (*m)[0][i2]  +  \
    //                      C1x * ((*m)[1][i2]) + \
    //                      C2x * ((*m)[2][i2]) + \
    //                      C3x * ((*m)[3][i2]) + \
    //                      C4x * ((*m)[4][i2]) + \
    //                      C5x *   ((*m)[5][i2]);
    //       (*d)[n3 - 1][i2] += C0x * (*m)[n3 - 1][i2]  +  \
    //                           C1x * ((*m)[n3 - 2][i2]) + \
    //                           C2x * ((*m)[n3 - 3][i2]) + \
    //                           C3x * ((*m)[n3 - 4][i2]) + \
    //                           C4x * ((*m)[n3 - 5][i2]) + \
    //                           C5x * ((*m)[n3 - 6][i2]);
    //     }
    //     else if (i3 == 1) {
    //       (*d)[1][i2] += C0x * (*m)[1][i2]  +                 \
    //                      C1x * ((*m)[2][i2]  + (*m)[0][i2]) + \
    //                      C2x * ((*m)[3][i2]) +                \
    //                      C3x * ((*m)[4][i2]) +                \
    //                      C4x * ((*m)[5][i2]) +                \
    //                      C5x *   ((*m)[6][i2]);
    //       (*d)[n3 - 2][i2] += C0x * (*m)[n3 - 2][i2]  + \
    //                           C1x *
    //                           ((*m)[n3 - 3][i2]  +
    //                            (*m)[n3 - 1][i2]) +       \
    //                           C2x * ((*m)[n3 - 4][i2]) + \
    //                           C3x * ((*m)[n3 - 5][i2]) + \
    //                           C4x * ((*m)[n3 - 6][i2]) + \
    //                           C5x * ((*m)[n3 - 7][i2]);
    //     }
    //     else if (i3 == 2) {
    //       (*d)[2][i2] += C0x * (*m)[2][i2]  + \
    //                      C1x *
    //                      ((*m)[1][i2]  + (*m)[3][i2]) + \
    //                      C2x *
    //                      ((*m)[0][i2]  + (*m)[4][i2]) + \
    //                      C3x * ((*m)[5][i2]) +          \
    //                      C4x * ((*m)[6][i2]) +          \
    //                      C5x *   ((*m)[7][i2]);
    //       (*d)[n3 - 3][i2] += C0x * (*m)[n3 - 3][i2]  + \
    //                           C1x *
    //                           ((*m)[n3 - 4][i2]  +
    //                            (*m)[n3 - 2][i2]) + \
    //                           C2x *
    //                           ((*m)[n3 - 5][i2]  +
    //                            (*m)[n3 - 1][i2]) +       \
    //                           C3x * ((*m)[n3 - 6][i2]) + \
    //                           C4x * ((*m)[n3 - 7][i2]) + \
    //                           C5x * ((*m)[n3 - 8][i2]);
    //     }
    //     else if (i3 == 3) {
    //       (*d)[3][i2] += C0x * (*m)[3][i2]  + \
    //                      C1x *
    //                      ((*m)[2][i2]  + (*m)[4][i2]) + \
    //                      C2x *
    //                      ((*m)[1][i2]  + (*m)[5][i2]) + \
    //                      C3x *
    //                      ((*m)[0][i2]  + (*m)[6][i2]) + \
    //                      C4x * ((*m)[7][i2]) +          \
    //                      C5x * ((*m)[8][i2]);
    //       (*d)[n3 - 4][i2] += C0x * (*m)[n3 - 4][i2]  + \
    //                           C1x *
    //                           ((*m)[n3 - 5][i2]  +
    //                            (*m)[n3 - 3][i2]) + \
    //                           C2x *
    //                           ((*m)[n3 - 6][i2]  +
    //                            (*m)[n3 - 2][i2]) + \
    //                           C3x *
    //                           ((*m)[n3 - 7][i2]  +
    //                            (*m)[n3 - 1][i2]) +       \
    //                           C4x * ((*m)[n3 - 8][i2]) + \
    //                           C5x * ((*m)[n3 - 9][i2]);
    //     }
    //     else if (i3 == 4) {
    //       (*d)[4][i2] += C0x * (*m)[4][i2]  + \
    //                      C1x *
    //                      ((*m)[3][i2]  + (*m)[5][i2]) + \
    //                      C2x *
    //                      ((*m)[2][i2]  + (*m)[6][i2]) + \
    //                      C3x *
    //                      ((*m)[1][i2]  + (*m)[7][i2]) + \
    //                      C4x *
    //                      ((*m)[0][i2]  + (*m)[8][i2]) + \
    //                      C5x *   ((*m)[9][i2]);
    //       (*d)[n3 - 5][i2] += C0x * (*m)[n3 - 5][i2]  + \
    //                           C1x *
    //                           ((*m)[n3 - 6][i2]  +
    //                            (*m)[n3 - 4][i2]) + \
    //                           C2x *
    //                           ((*m)[n3 - 7][i2]  +
    //                            (*m)[n3 - 3][i2]) + \
    //                           C3x *
    //                           ((*m)[n3 - 8][i2]  +
    //                            (*m)[n3 - 2][i2]) + \
    //                           C4x *
    //                           ((*m)[n3 - 9][i2]  +
    //                            (*m)[n3 - 1][i2]) + \
    //                           C5x *   ((*m)[n3 - 10][i2]);
    //     }
    //   }
    // }
    //
    // for (i3 = 0; i3 < n3; i3++) {
    //   for (i2 = 0; i2 < _bufferSize; i2++) {
    //     if (i2 == 0) {
    //       (*d)[i3][0] += C0z * (*m)[i3][0]  +  \
    //                      C1z * ((*m)[i3][1]) + \
    //                      C2z * ((*m)[i3][2]) + \
    //                      C3z * ((*m)[i3][3]) + \
    //                      C4z * ((*m)[i3][4]) + \
    //                      C5z * ((*m)[i3][5]);
    //       (*d)[i3][n2 - 1] += C0z * (*m)[i3][n2 - 1]  +  \
    //                           C1z * ((*m)[i3][n2 - 2]) + \
    //                           C2z * ((*m)[i3][n2 - 3]) + \
    //                           C3z * ((*m)[i3][n2 - 4]) + \
    //                           C4z * ((*m)[i3][n2 - 5]) + \
    //                           C5z * ((*m)[i3][n2 - 6]);
    //     }
    //     else if (i2 == 1) {
    //       (*d)[i3][1] += C0z * (*m)[i3][1]  + \
    //                      C1z *
    //                      ((*m)[i3][0]  + (*m)[i3][2]) + \
    //                      C2z * ((*m)[i3][3]) +          \
    //                      C3z * ((*m)[i3][4]) +          \
    //                      C4z * ((*m)[i3][5]) +          \
    //                      C5z * ((*m)[i3][6]);
    //       (*d)[i3][n2 - 2] += C0z * (*m)[i3][n2 - 2]  + \
    //                           C1z *
    //                           ((*m)[i3][n2 - 3]  +
    //                            (*m)[i3][n2 - 1]) +       \
    //                           C2z * ((*m)[i3][n2 - 4]) + \
    //                           C3z * ((*m)[i3][n2 - 5]) + \
    //                           C4z * ((*m)[i3][n2 - 6]) + \
    //                           C5z * ((*m)[i3][n2 - 7]);
    //     }
    //     else if (i2 == 2) {
    //       (*d)[i3][2] += C0z * (*m)[i3][2]  + \
    //                      C1z *
    //                      ((*m)[i3][1]  + (*m)[i3][3]) + \
    //                      C2z *
    //                      ((*m)[i3][0]  + (*m)[i3][4]) + \
    //                      C3z * ((*m)[i3][5]) +          \
    //                      C4z * ((*m)[i3][6]) +          \
    //                      C5z * ((*m)[i3][7]);
    //       (*d)[i3][n2 - 3] += C0z * (*m)[i3][n2 - 3]  + \
    //                           C1z *
    //                           ((*m)[i3][n2 - 4]  +
    //                            (*m)[i3][n2 - 2]) + \
    //                           C2z *
    //                           ((*m)[i3][n2 - 5]  +
    //                            (*m)[i3][n2 - 1]) +       \
    //                           C3z * ((*m)[i3][n2 - 6]) + \
    //                           C4z * ((*m)[i3][n2 - 7]) + \
    //                           C5z * ((*m)[i3][n2 - 8]);
    //     }
    //     else if (i2 == 3) {
    //       (*d)[i3][3] += C0z * (*m)[i3][3]  + \
    //                      C1z *
    //                      ((*m)[i3][2]  + (*m)[i3][4]) + \
    //                      C2z *
    //                      ((*m)[i3][1]  + (*m)[i3][5]) + \
    //                      C3z *
    //                      ((*m)[i3][0]  + (*m)[i3][6]) + \
    //                      C4z * ((*m)[i3][7]) +          \
    //                      C5z * ((*m)[i3][8]);
    //       (*d)[i3][n2 - 4] += C0z * (*m)[i3][n2 - 4]  + \
    //                           C1z *
    //                           ((*m)[i3][n2 - 5]  +
    //                            (*m)[i3][n2 - 3]) + \
    //                           C2z *
    //                           ((*m)[i3][n2 - 6]  +
    //                            (*m)[i3][n2 - 2]) + \
    //                           C3z *
    //                           ((*m)[i3][n2 - 7]  +
    //                            (*m)[i3][n2 - 1]) +       \
    //                           C4z * ((*m)[i3][n2 - 8]) + \
    //                           C5z * ((*m)[i3][n2 - 9]);
    //     }
    //     else if (i2 == 4) {
    //       (*d)[i3][4] += C0z * (*m)[i3][4]  + \
    //                      C1z *
    //                      ((*m)[i3][3]  + (*m)[i3][5]) + \
    //                      C2z *
    //                      ((*m)[i3][2]  + (*m)[i3][6]) + \
    //                      C3z *
    //                      ((*m)[i3][1]  + (*m)[i3][7]) + \
    //                      C4z *
    //                      ((*m)[i3][0]  + (*m)[i3][8]) + \
    //                      C5z *   ((*m)[i3][9]);
    //       (*d)[i3][n2 - 5] += C0z * (*m)[i3][n2 - 5]  + \
    //                           C1z *
    //                           ((*m)[i3][n2 - 6]  +
    //                            (*m)[i3][n2 - 4]) + \
    //                           C2z *
    //                           ((*m)[i3][n2 - 7]  +
    //                            (*m)[i3][n2 - 3]) + \
    //                           C3z *
    //                           ((*m)[i3][n2 - 8]  +
    //                            (*m)[i3][n2 - 2]) + \
    //                           C4z *
    //                           ((*m)[i3][n2 - 9]  +
    //                            (*m)[i3][n2 - 1]) + \
    //                           C5z *   ((*m)[i3][n2 - 10]);
    //     }
    //   }
    // }

    // calculate laplacian.
    #pragma omp parallel for collapse(2)

    for (i3 = _bufferSize; i3 < n3 - _bufferSize; i3++) {
      for (i2 = _bufferSize; i2 < n2 - _bufferSize; i2++) {
        (*d)[i3][i2] +=       C0x *
                        (*m)[i3][i2]  + \
                        C1x *
                        ((*m)[i3 + 1][i2]  +
                         (*m)[i3 - 1][i2]) + \
                        C2x *
                        ((*m)[i3 + 2][i2]  +
                         (*m)[i3 - 2][i2]) + \
                        C3x *
                        ((*m)[i3 + 3][i2]  +
                         (*m)[i3 - 3][i2]) + \
                        C4x *
                        ((*m)[i3 + 4][i2]  +
                         (*m)[i3 - 4][i2]) + \
                        C5x *
                        ((*m)[i3 + 5][i2]  +
                         (*m)[i3 - 5][i2]) + \
                        C0z *
                        (*m)[i3][i2]  + \
                        C1z *
                        ((*m)[i3][i2 + 1]  +
                         (*m)[i3][i2 - 1]) + \
                        C2z *
                        ((*m)[i3][i2 + 2]  +
                         (*m)[i3][i2 - 2]) + \
                        C3z *
                        ((*m)[i3][i2 + 3]  +
                         (*m)[i3][i2 - 3]) + \
                        C4z *
                        ((*m)[i3][i2 + 4]  +
                         (*m)[i3][i2 - 4]) + \
                        C5z *
                        ((*m)[i3][i2 + 5]  +
                         (*m)[i3][i2 - 5]);
      }
    }
  }
}

void Laplacian2dFast::adjoint(const bool                         add,
                              std::shared_ptr<giee::Vector>      model,
                              const std::shared_ptr<giee::Vector>data) {
  assert(checkDomainRange(model, data, true));

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
    std::shared_ptr<float3DReg> buffer(new  float3DReg(n1,
                                                       n2 + 2 * _bufferSize,
                                                       n3 + 2 * _bufferSize));

    // buffer->scale(0);
    // std::shared_ptr<float3D> b =
    //   ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);
    //
    //
    // #pragma omp parallel for collapse(3)
    //
    // for (i3 = 0; i3 < n3; i3++) {
    //   for (i2 = 0; i2 < n2; i2++) {
    //     for (i1 = 0; i1 < n1; i1++) {
    //       (*b)[i3 + _bufferSize][i2 + _bufferSize][i1] = (*r)[i3][i2][i1];
    //     }
    //   }
    // }


    for (i3 = 0; i3 < _bufferSize; i3++) {
      for (i2 = 0; i2 < n2; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          // (*m)[i3][i2][i1] += C0z * (*b)[i3][i2][i1] + C0x *
          // (*b)[i3][i2][i1];
          if (i3 == 0) {
            (*m)[0][i2][i1] += C0x * (*d)[0][i2][i1] +   \
                               C1x * ((*d)[1][i2][i1]) + \
                               C2x * ((*d)[2][i2][i1]) + \
                               C3x * ((*d)[3][i2][i1]) + \
                               C4x * ((*d)[4][i2][i1]) + \
                               C5x *   ((*d)[5][i2][i1]);
            (*m)[n3 - 1][i2][i1] += C0x * (*d)[n3 - 1][i2][i1] +   \
                                    C1x * ((*d)[n3 - 2][i2][i1]) + \
                                    C2x * ((*d)[n3 - 3][i2][i1]) + \
                                    C3x * ((*d)[n3 - 4][i2][i1]) + \
                                    C4x * ((*d)[n3 - 5][i2][i1]) + \
                                    C5x * ((*d)[n3 - 6][i2][i1]);
          }
          else if (i3 == 1) {
            (*m)[1][i2][i1] += C0x * (*d)[1][i2][i1] +                     \
                               C1x * ((*d)[2][i2][i1] + (*d)[0][i2][i1]) + \
                               C2x * ((*d)[3][i2][i1]) +                   \
                               C3x * ((*d)[4][i2][i1]) +                   \
                               C4x * ((*d)[5][i2][i1]) +                   \
                               C5x *   ((*d)[6][i2][i1]);
            (*m)[n3 - 2][i2][i1] += C0x * (*d)[n3 - 2][i2][i1] + \
                                    C1x *
                                    ((*d)[n3 - 3][i2][i1] +
                                     (*d)[n3 - 1][i2][i1]) +       \
                                    C2x * ((*d)[n3 - 4][i2][i1]) + \
                                    C3x * ((*d)[n3 - 5][i2][i1]) + \
                                    C4x * ((*d)[n3 - 6][i2][i1]) + \
                                    C5x * ((*d)[n3 - 7][i2][i1]);
          }
          else if (i3 == 2) {
            (*m)[2][i2][i1] += C0x * (*d)[2][i2][i1] + \
                               C1x *
                               ((*d)[1][i2][i1] + (*d)[3][i2][i1]) + \
                               C2x *
                               ((*d)[0][i2][i1] + (*d)[4][i2][i1]) + \
                               C3x * ((*d)[5][i2][i1]) +             \
                               C4x * ((*d)[6][i2][i1]) +             \
                               C5x *   ((*d)[7][i2][i1]);
            (*m)[n3 - 3][i2][i1] += C0x * (*d)[n3 - 3][i2][i1] + \
                                    C1x *
                                    ((*d)[n3 - 4][i2][i1] +
                                     (*d)[n3 - 2][i2][i1]) + \
                                    C2x *
                                    ((*d)[n3 - 5][i2][i1] +
                                     (*d)[n3 - 1][i2][i1]) +       \
                                    C3x * ((*d)[n3 - 6][i2][i1]) + \
                                    C4x * ((*d)[n3 - 7][i2][i1]) + \
                                    C5x * ((*d)[n3 - 8][i2][i1]);
          }
          else if (i3 == 3) {
            (*m)[3][i2][i1] += C0x * (*d)[i3][i2][i1] + \
                               C1x *
                               ((*d)[2][i2][i1] + (*d)[4][i2][i1]) + \
                               C2x *
                               ((*d)[1][i2][i1] + (*d)[5][i2][i1]) + \
                               C3x *
                               ((*d)[0][i2][i1] + (*d)[6][i2][i1]) + \
                               C4x * ((*d)[7][i2][i1]) +             \
                               C5x * ((*d)[8][i2][i1]);
            (*m)[n3 - 4][i2][i1] += C0x * (*d)[n3 - 4][i2][i1] + \
                                    C1x *
                                    ((*d)[n3 - 5][i2][i1] +
                                     (*d)[n3 - 3][i2][i1]) + \
                                    C2x *
                                    ((*d)[n3 - 6][i2][i1] +
                                     (*d)[n3 - 2][i2][i1]) + \
                                    C3x *
                                    ((*d)[n3 - 7][i2][i1] +
                                     (*d)[n3 - 1][i2][i1]) +       \
                                    C4x * ((*d)[n3 - 8][i2][i1]) + \
                                    C5x * ((*d)[n3 - 9][i2][i1]);
          }
          else if (i3 == 4) {
            (*m)[4][i2][i1] += C0x * (*d)[4][i2][i1] + \
                               C1x *
                               ((*d)[3][i2][i1] + (*d)[5][i2][i1]) + \
                               C2x *
                               ((*d)[2][i2][i1] + (*d)[6][i2][i1]) + \
                               C3x *
                               ((*d)[1][i2][i1] + (*d)[7][i2][i1]) + \
                               C4x *
                               ((*d)[0][i2][i1] + (*d)[8][i2][i1]) + \
                               C5x *   ((*d)[9][i2][i1]);
            (*m)[n3 - 5][i2][i1] += C0x * (*d)[n3 - 5][i2][i1] + \
                                    C1x *
                                    ((*d)[n3 - 6][i2][i1] +
                                     (*d)[n3 - 4][i2][i1]) + \
                                    C2x *
                                    ((*d)[n3 - 7][i2][i1] +
                                     (*d)[n3 - 3][i2][i1]) + \
                                    C3x *
                                    ((*d)[n3 - 8][i2][i1] +
                                     (*d)[n3 - 2][i2][i1]) + \
                                    C4x *
                                    ((*d)[n3 - 9][i2][i1] +
                                     (*d)[n3 - 1][i2][i1]) + \
                                    C5x *   ((*d)[n3 - 10][i2][i1]);
          }
        }
      }
    }

    for (i3 = 0; i3 < n3; i3++) {
      for (i2 = 0; i2 < _bufferSize; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          if (i2 == 0) {
            (*m)[i3][0][i1] += C0z * (*d)[i3][0][i1] +   \
                               C1z * ((*d)[i3][1][i1]) + \
                               C2z * ((*d)[i3][2][i1]) + \
                               C3z * ((*d)[i3][3][i1]) + \
                               C4z * ((*d)[i3][4][i1]) + \
                               C5z * ((*d)[i3][5][i1]);
            (*m)[i3][n2 - 1][i1] += C0z * (*d)[i3][n2 - 1][i1] +   \
                                    C1z * ((*d)[i3][n2 - 2][i1]) + \
                                    C2z * ((*d)[i3][n2 - 3][i1]) + \
                                    C3z * ((*d)[i3][n2 - 4][i1]) + \
                                    C4z * ((*d)[i3][n2 - 5][i1]) + \
                                    C5z * ((*d)[i3][n2 - 6][i1]);
          }
          else if (i2 == 1) {
            (*m)[i3][1][i1] += C0z * (*d)[i3][i2][i1] + \
                               C1z *
                               ((*d)[i3][i2 + 1][i1] + (*d)[i3][i2 - 1][i1]) + \
                               C2z * ((*d)[i3][i2 + 2][i1]) +                  \
                               C3z * ((*d)[i3][i2 + 3][i1]) +                  \
                               C4z * ((*d)[i3][i2 + 4][i1]) +                  \
                               C5z * ((*d)[i3][i2 + 5][i1]);
            (*m)[i3][n2 - 2][i1] += C0z * (*d)[i3][n2 - 2][i1] + \
                                    C1z *
                                    ((*d)[i3][n2 - 3][i1] +
                                     (*d)[i3][n2 - 1][i1]) +       \
                                    C2z * ((*d)[i3][n2 - 4][i1]) + \
                                    C3z * ((*d)[i3][n2 - 5][i1]) + \
                                    C4z * ((*d)[i3][n2 - 6][i1]) + \
                                    C5z * ((*d)[i3][n2 - 7][i1]);
          }
          else if (i2 == 2) {
            (*m)[i3][2][i1] += C0z * (*d)[i3][2][i1] + \
                               C1z *
                               ((*d)[i3][1][i1] + (*d)[i3][3][i1]) + \
                               C2z *
                               ((*d)[i3][0][i1] + (*d)[i3][4][i1]) + \
                               C3z * ((*d)[i3][5][i1]) +             \
                               C4z * ((*d)[i3][6][i1]) +             \
                               C5z * ((*d)[i3][7][i1]);
            (*m)[i3][n2 - 3][i1] += C0z * (*d)[i3][n2 - 3][i1] + \
                                    C1z *
                                    ((*d)[i3][n2 - 4][i1] +
                                     (*d)[i3][n2 - 2][i1]) + \
                                    C2z *
                                    ((*d)[i3][n2 - 5][i1] +
                                     (*d)[i3][n2 - 1][i1]) +       \
                                    C3z * ((*d)[i3][n2 - 6][i1]) + \
                                    C4z * ((*d)[i3][n2 - 7][i1]) + \
                                    C5z * ((*d)[i3][n2 - 8][i1]);
          }
          else if (i2 == 3) {
            (*m)[i3][3][i1] += C0z * (*d)[i3][3][i1] + \
                               C1z *
                               ((*d)[i3][2][i1] + (*d)[i3][4][i1]) + \
                               C2z *
                               ((*d)[i3][1][i1] + (*d)[i3][5][i1]) + \
                               C3z *
                               ((*d)[i3][0][i1] + (*d)[i3][6][i1]) + \
                               C4z * ((*d)[i3][7][i1]) +             \
                               C5z * ((*d)[i3][8][i1]);
            (*m)[i3][n2 - 4][i1] += C0z * (*d)[i3][n2 - 4][i1] + \
                                    C1z *
                                    ((*d)[i3][n2 - 5][i1] +
                                     (*d)[i3][n2 - 3][i1]) + \
                                    C2z *
                                    ((*d)[i3][n2 - 6][i1] +
                                     (*d)[i3][n2 - 2][i1]) + \
                                    C3z *
                                    ((*d)[i3][n2 - 7][i1] +
                                     (*d)[i3][n2 - 1][i1]) +       \
                                    C4z * ((*d)[i3][n2 - 8][i1]) + \
                                    C5z * ((*d)[i3][n2 - 9][i1]);
          }
          else if (i2 == 4) {
            (*m)[i3][4][i1] += C0z * (*d)[i3][4][i1] + \
                               C1z *
                               ((*d)[i3][3][i1] + (*d)[i3][5][i1]) + \
                               C2z *
                               ((*d)[i3][2][i1] + (*d)[i3][6][i1]) + \
                               C3z *
                               ((*d)[i3][1][i1] + (*d)[i3][7][i1]) + \
                               C4z *
                               ((*d)[i3][0][i1] + (*d)[i3][8][i1]) + \
                               C5z *   ((*d)[i3][9][i1]);
            (*m)[i3][n2 - 5][i1] += C0z * (*d)[i3][n2 - 5][i1] + \
                                    C1z *
                                    ((*d)[i3][n2 - 6][i1] +
                                     (*d)[i3][n2 - 4][i1]) + \
                                    C2z *
                                    ((*d)[i3][n2 - 7][i1] +
                                     (*d)[i3][n2 - 3][i1]) + \
                                    C3z *
                                    ((*d)[i3][n2 - 8][i1] +
                                     (*d)[i3][n2 - 2][i1]) + \
                                    C4z *
                                    ((*d)[i3][n2 - 9][i1] +
                                     (*d)[i3][n2 - 1][i1]) + \
                                    C5z *   ((*d)[i3][n2 - 10][i1]);
          }
        }
      }
    }

    // calculate laplacian.
    #pragma omp parallel for collapse(3)

    for (i3 = _bufferSize; i3 < n3 - _bufferSize; i3++) {
      for (i2 = _bufferSize; i2 < n2 - _bufferSize; i2++) {
        for (i1 = 0; i1 < n1; i1++) {
          (*m)[i3][i2][i1] +=       C0x *
                              (*d)[i3][i2][i1] + \
                              C1x *
                              ((*d)[i3 + 1][i2][i1] +
                               (*d)[i3 - 1][i2][i1]) + \
                              C2x *
                              ((*d)[i3 + 2][i2][i1] +
                               (*d)[i3 - 2][i2][i1]) + \
                              C3x *
                              ((*d)[i3 + 3][i2][i1] +
                               (*d)[i3 - 3][i2][i1]) + \
                              C4x *
                              ((*d)[i3 + 4][i2][i1] +
                               (*d)[i3 - 4][i2][i1]) + \
                              C5x *
                              ((*d)[i3 + 5][i2][i1] +
                               (*d)[i3 - 5][i2][i1]) + \
                              C0z *
                              (*d)[i3][i2][i1] + \
                              C1z *
                              ((*d)[i3][i2 + 1][i1] +
                               (*d)[i3][i2 - 1][i1]) + \
                              C2z *
                              ((*d)[i3][i2 + 2][i1] +
                               (*d)[i3][i2 - 2][i1]) + \
                              C3z *
                              ((*d)[i3][i2 + 3][i1] +
                               (*d)[i3][i2 - 3][i1]) + \
                              C4z *
                              ((*d)[i3][i2 + 4][i1] +
                               (*d)[i3][i2 - 4][i1]) + \
                              C5z *
                              ((*d)[i3][i2 + 5][i1] +
                               (*d)[i3][i2 - 5][i1]);
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
    // buffer->scale(0);
    // std::shared_ptr<float2D> b =
    //   ((std::dynamic_pointer_cast<float2DReg>(buffer))->_mat);
    //
    // #pragma omp parallel for collapse(2)
    //
    // for (i3 = 0; i3 < n3; i3++) {
    //   for (i2 = 0; i2 < n2; i2++) {
    //     (*b)[i3 + _bufferSize][i2 + _bufferSize] = (*d)[i3][i2];
    //   }
    // }

    // for (i3 = 0; i3 < _bufferSize; i3++) {
    //   for (i2 = 0; i2 < n2; i2++) {
    //     // ( *m)[i3][i2]  += C0z * (*b)[i3][i2]  + C0x *
    //     // (*b)[i3][i2] ;
    //     if (i3 == 0) {
    //       (*m)[0][i2] += C0x * (*d)[0][i2]  +  \
    //                      C1x * ((*d)[1][i2]) + \
    //                      C2x * ((*d)[2][i2]) + \
    //                      C3x * ((*d)[3][i2]) + \
    //                      C4x * ((*d)[4][i2]) + \
    //                      C5x *   ((*d)[5][i2]);
    //       (*m)[n3 - 1][i2] += C0x * (*d)[n3 - 1][i2]  +  \
    //                           C1x * ((*d)[n3 - 2][i2]) + \
    //                           C2x * ((*d)[n3 - 3][i2]) + \
    //                           C3x * ((*d)[n3 - 4][i2]) + \
    //                           C4x * ((*d)[n3 - 5][i2]) + \
    //                           C5x * ((*d)[n3 - 6][i2]);
    //     }
    //     else if (i3 == 1) {
    //       (*m)[1][i2] += C0x * (*d)[1][i2]  +                 \
    //                      C1x * ((*d)[2][i2]  + (*d)[0][i2]) + \
    //                      C2x * ((*d)[3][i2]) +                \
    //                      C3x * ((*d)[4][i2]) +                \
    //                      C4x * ((*d)[5][i2]) +                \
    //                      C5x *   ((*d)[6][i2]);
    //       (*m)[n3 - 2][i2] += C0x * (*d)[n3 - 2][i2]  + \
    //                           C1x *
    //                           ((*d)[n3 - 3][i2]  +
    //                            (*d)[n3 - 1][i2]) +       \
    //                           C2x * ((*d)[n3 - 4][i2]) + \
    //                           C3x * ((*d)[n3 - 5][i2]) + \
    //                           C4x * ((*d)[n3 - 6][i2]) + \
    //                           C5x * ((*d)[n3 - 7][i2]);
    //     }
    //     else if (i3 == 2) {
    //       (*m)[2][i2] += C0x * (*d)[2][i2]  + \
    //                      C1x *
    //                      ((*d)[1][i2]  + (*d)[3][i2]) + \
    //                      C2x *
    //                      ((*d)[0][i2]  + (*d)[4][i2]) + \
    //                      C3x * ((*d)[5][i2]) +          \
    //                      C4x * ((*d)[6][i2]) +          \
    //                      C5x *   ((*d)[7][i2]);
    //       (*m)[n3 - 3][i2] += C0x * (*d)[n3 - 3][i2]  + \
    //                           C1x *
    //                           ((*d)[n3 - 4][i2]  +
    //                            (*d)[n3 - 2][i2]) + \
    //                           C2x *
    //                           ((*d)[n3 - 5][i2]  +
    //                            (*d)[n3 - 1][i2]) +       \
    //                           C3x * ((*d)[n3 - 6][i2]) + \
    //                           C4x * ((*d)[n3 - 7][i2]) + \
    //                           C5x * ((*d)[n3 - 8][i2]);
    //     }
    //     else if (i3 == 3) {
    //       (*m)[3][i2] += C0x * (*d)[i3][i2]  + \
    //                      C1x *
    //                      ((*d)[2][i2]  + (*d)[4][i2]) + \
    //                      C2x *
    //                      ((*d)[1][i2]  + (*d)[5][i2]) + \
    //                      C3x *
    //                      ((*d)[0][i2]  + (*d)[6][i2]) + \
    //                      C4x * ((*d)[7][i2]) +          \
    //                      C5x * ((*d)[8][i2]);
    //       (*m)[n3 - 4][i2] += C0x * (*d)[n3 - 4][i2]  + \
    //                           C1x *
    //                           ((*d)[n3 - 5][i2]  +
    //                            (*d)[n3 - 3][i2]) + \
    //                           C2x *
    //                           ((*d)[n3 - 6][i2]  +
    //                            (*d)[n3 - 2][i2]) + \
    //                           C3x *
    //                           ((*d)[n3 - 7][i2]  +
    //                            (*d)[n3 - 1][i2]) +       \
    //                           C4x * ((*d)[n3 - 8][i2]) + \
    //                           C5x * ((*d)[n3 - 9][i2]);
    //     }
    //     else if (i3 == 4) {
    //       (*m)[4][i2] += C0x * (*d)[4][i2]  + \
    //                      C1x *
    //                      ((*d)[3][i2]  + (*d)[5][i2]) + \
    //                      C2x *
    //                      ((*d)[2][i2]  + (*d)[6][i2]) + \
    //                      C3x *
    //                      ((*d)[1][i2]  + (*d)[7][i2]) + \
    //                      C4x *
    //                      ((*d)[0][i2]  + (*d)[8][i2]) + \
    //                      C5x *   ((*d)[9][i2]);
    //       (*m)[n3 - 5][i2] += C0x * (*d)[n3 - 5][i2]  + \
    //                           C1x *
    //                           ((*d)[n3 - 6][i2]  +
    //                            (*d)[n3 - 4][i2]) + \
    //                           C2x *
    //                           ((*d)[n3 - 7][i2]  +
    //                            (*d)[n3 - 3][i2]) + \
    //                           C3x *
    //                           ((*d)[n3 - 8][i2]  +
    //                            (*d)[n3 - 2][i2]) + \
    //                           C4x *
    //                           ((*d)[n3 - 9][i2]  +
    //                            (*d)[n3 - 1][i2]) + \
    //                           C5x *   ((*d)[n3 - 10][i2]);
    //     }
    //   }
    // }
    //
    // for (i3 = 0; i3 < n3; i3++) {
    //   for (i2 = 0; i2 < _bufferSize; i2++) {
    //     if (i2 == 0) {
    //       (*m)[i3][0] += C0z * (*d)[i3][0]  +  \
    //                      C1z * ((*d)[i3][1]) + \
    //                      C2z * ((*d)[i3][2]) + \
    //                      C3z * ((*d)[i3][3]) + \
    //                      C4z * ((*d)[i3][4]) + \
    //                      C5z * ((*d)[i3][5]);
    //       (*m)[i3][n2 - 1] += C0z * (*d)[i3][n2 - 1]  +  \
    //                           C1z * ((*d)[i3][n2 - 2]) + \
    //                           C2z * ((*d)[i3][n2 - 3]) + \
    //                           C3z * ((*d)[i3][n2 - 4]) + \
    //                           C4z * ((*d)[i3][n2 - 5]) + \
    //                           C5z * ((*d)[i3][n2 - 6]);
    //     }
    //     else if (i2 == 1) {
    //       (*m)[i3][1] += C0z * (*d)[i3][1]  + \
    //                      C1z *
    //                      ((*d)[i3][0]  + (*d)[i3][2]) + \
    //                      C2z * ((*d)[i3][3]) +          \
    //                      C3z * ((*d)[i3][4]) +          \
    //                      C4z * ((*d)[i3][5]) +          \
    //                      C5z * ((*d)[i3][6]);
    //       (*m)[i3][n2 - 2] += C0z * (*d)[i3][n2 - 2]  + \
    //                           C1z *
    //                           ((*d)[i3][n2 - 3]  +
    //                            (*d)[i3][n2 - 1]) +       \
    //                           C2z * ((*d)[i3][n2 - 4]) + \
    //                           C3z * ((*d)[i3][n2 - 5]) + \
    //                           C4z * ((*d)[i3][n2 - 6]) + \
    //                           C5z * ((*d)[i3][n2 - 7]);
    //     }
    //     else if (i2 == 2) {
    //       (*m)[i3][2] += C0z * (*d)[i3][2]  + \
    //                      C1z *
    //                      ((*d)[i3][1]  + (*d)[i3][3]) + \
    //                      C2z *
    //                      ((*d)[i3][0]  + (*d)[i3][4]) + \
    //                      C3z * ((*d)[i3][5]) +          \
    //                      C4z * ((*d)[i3][6]) +          \
    //                      C5z * ((*d)[i3][7]);
    //       (*m)[i3][n2 - 3] += C0z * (*d)[i3][n2 - 3]  + \
    //                           C1z *
    //                           ((*d)[i3][n2 - 4]  +
    //                            (*d)[i3][n2 - 2]) + \
    //                           C2z *
    //                           ((*d)[i3][n2 - 5]  +
    //                            (*d)[i3][n2 - 1]) +       \
    //                           C3z * ((*d)[i3][n2 - 6]) + \
    //                           C4z * ((*d)[i3][n2 - 7]) + \
    //                           C5z * ((*d)[i3][n2 - 8]);
    //     }
    //     else if (i2 == 3) {
    //       (*m)[i3][3] += C0z * (*d)[i3][3]  + \
    //                      C1z *
    //                      ((*d)[i3][2]  + (*d)[i3][4]) + \
    //                      C2z *
    //                      ((*d)[i3][1]  + (*d)[i3][5]) + \
    //                      C3z *
    //                      ((*d)[i3][0]  + (*d)[i3][6]) + \
    //                      C4z * ((*d)[i3][7]) +          \
    //                      C5z * ((*d)[i3][8]);
    //       (*m)[i3][n2 - 4] += C0z * (*d)[i3][n2 - 4]  + \
    //                           C1z *
    //                           ((*d)[i3][n2 - 5]  +
    //                            (*d)[i3][n2 - 3]) + \
    //                           C2z *
    //                           ((*d)[i3][n2 - 6]  +
    //                            (*d)[i3][n2 - 2]) + \
    //                           C3z *
    //                           ((*d)[i3][n2 - 7]  +
    //                            (*d)[i3][n2 - 1]) +       \
    //                           C4z * ((*d)[i3][n2 - 8]) + \
    //                           C5z * ((*d)[i3][n2 - 9]);
    //     }
    //     else if (i2 == 4) {
    //       (*m)[i3][4] += C0z * (*d)[i3][4]  + \
    //                      C1z *
    //                      ((*d)[i3][3]  + (*d)[i3][5]) + \
    //                      C2z *
    //                      ((*d)[i3][2]  + (*d)[i3][6]) + \
    //                      C3z *
    //                      ((*d)[i3][1]  + (*d)[i3][7]) + \
    //                      C4z *
    //                      ((*d)[i3][0]  + (*d)[i3][8]) + \
    //                      C5z *   ((*d)[i3][9]);
    //       (*m)[i3][n2 - 5] += C0z * (*d)[i3][n2 - 5]  + \
    //                           C1z *
    //                           ((*d)[i3][n2 - 6]  +
    //                            (*d)[i3][n2 - 4]) + \
    //                           C2z *
    //                           ((*d)[i3][n2 - 7]  +
    //                            (*d)[i3][n2 - 3]) + \
    //                           C3z *
    //                           ((*d)[i3][n2 - 8]  +
    //                            (*d)[i3][n2 - 2]) + \
    //                           C4z *
    //                           ((*d)[i3][n2 - 9]  +
    //                            (*d)[i3][n2 - 1]) + \
    //                           C5z *   ((*d)[i3][n2 - 10]);
    //     }
    //   }
    // }

    // calculate laplacian.
    #pragma omp parallel for collapse(2)

    for (i3 = _bufferSize; i3 < n3 - _bufferSize; i3++) {
      for (i2 = _bufferSize; i2 < n2 - _bufferSize; i2++) {
        (*m)[i3][i2] +=       C0x *
                        (*d)[i3][i2]  + \
                        C1x *
                        ((*d)[i3 + 1][i2]  +
                         (*d)[i3 - 1][i2]) + \
                        C2x *
                        ((*d)[i3 + 2][i2]  +
                         (*d)[i3 - 2][i2]) + \
                        C3x *
                        ((*d)[i3 + 3][i2]  +
                         (*d)[i3 - 3][i2]) + \
                        C4x *
                        ((*d)[i3 + 4][i2]  +
                         (*d)[i3 - 4][i2]) + \
                        C5x *
                        ((*d)[i3 + 5][i2]  +
                         (*d)[i3 - 5][i2]) + \
                        C0z *
                        (*d)[i3][i2]  + \
                        C1z *
                        ((*d)[i3][i2 + 1]  +
                         (*d)[i3][i2 - 1]) + \
                        C2z *
                        ((*d)[i3][i2 + 2]  +
                         (*d)[i3][i2 - 2]) + \
                        C3z *
                        ((*d)[i3][i2 + 3]  +
                         (*d)[i3][i2 - 3]) + \
                        C4z *
                        ((*d)[i3][i2 + 4]  +
                         (*d)[i3][i2 - 4]) + \
                        C5z *
                        ((*d)[i3][i2 + 5]  +
                         (*d)[i3][i2 - 5]);
      }
    }
  }
}

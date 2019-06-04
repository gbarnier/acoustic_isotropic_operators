#include <PadModel3d.h>
using namespace giee;
using namespace waveform;
using namespace SEP;

PadModel3d::PadModel3d(
  const std::shared_ptr<giee::float3DReg>model,
  const std::shared_ptr<giee::float3DReg>data,
  const int                              padSize1,
  const int                              padSize2,
  const int                              padSize3,
  const int                              padOption)
{
  // model[3][2][1]

  // all data dimensions should be bigger by 2*padSize
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(
           1).n - 2 * padSize1);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(
           2).n - 2 * padSize2);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(
           3).n - 2 * padSize3);

  // spacing should match
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(
           1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(
           2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(
           3).d);

  // origin of data (ie padded model) axis should be smaller by padSize*d1
  assert(model->getHyper()->getAxis(1).o == data->getHyper()->getAxis(
           1).o + padSize1 * data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).o == data->getHyper()->getAxis(
           2).o + padSize2 * data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).o == data->getHyper()->getAxis(
           3).o + padSize3 * data->getHyper()->getAxis(3).d);

  // padOption should be 0 or 1
  assert(padOption == 0 || padOption == 1);

  // set domain and range
  setDomainRange(model, data);

  _padSize1 = padSize1;
  _padSize2 = padSize2;
  _padSize3 = padSize3;

  _padOption = padOption;
}

PadModel3d::PadModel3d(
  const std::shared_ptr<giee::float3DReg>model,
  const std::shared_ptr<giee::float3DReg>data,
  const int                              padSize,
  const int                              padOption)
{
  // model[3][2][1]

  // all data dimensions should be bigger by 2*padSize
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(
           1).n - 2 * padSize);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(
           2).n - 2 * padSize);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(
           3).n - 2 * padSize);

  // spacing should match
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(
           1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(
           2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(
           3).d);

  // origin of data (ie padded model) axis should be smaller by padSize*d1
  assert(model->getHyper()->getAxis(1).o == data->getHyper()->getAxis(
           1).o + padSize * data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).o == data->getHyper()->getAxis(
           2).o + padSize * data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).o == data->getHyper()->getAxis(
           3).o + padSize * data->getHyper()->getAxis(3).d);

  // padOption should be 0 or 1
  assert(padOption == 0 || padOption == 1);

  // set domain and range
  setDomainRange(model, data);

  _padSize1 = padSize;
  _padSize2 = padSize;
  _padSize3 = padSize;

  _padOption = padOption;
}

// adds padSize indices on either end of each axis
void PadModel3d::forward(const bool                         add,
                         const std::shared_ptr<giee::Vector>model,
                         std::shared_ptr<giee::Vector>      data) {
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);

  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  int n1d =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
  int n2d =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
  int n3d =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;
  int n1m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

  // corners
  for (int i3 = 0; i3 < _padSize3; i3++) {
    for (int i2 = 0; i2 < _padSize2; i2++) {
      for (int i1 = 0; i1 < _padSize1; i1++) {
        if (_padOption == 0) {
          (*d)[i3][i2][i1]                               += 0;
          (*d)[i3][n2d - i2 - 1][i1]                     += 0;
          (*d)[i3][i2][n1d - i1 - 1]                     += 0;
          (*d)[i3][n2d - i2 - 1][n1d - i1 - 1]           += 0;
          (*d)[n3d - i3 - 1][i2][i1]                     += 0;
          (*d)[n3d - i3 - 1][n2d - i2 - 1][i1]           += 0;
          (*d)[n3d - i3 - 1][i2][n1d - i1 - 1]           += 0;
          (*d)[n3d - i3 - 1][n2d - i2 - 1][n1d - i1 - 1] += 0;
        }
        else {
          (*d)[i3][i2][i1]           += (*m)[0][0][0];
          (*d)[i3][n2d - i2 - 1][i1] += (*m)[0][n2m - 1][0];
          (*d)[i3][i2][n1d - i1 - 1] += (*m)[0][0][n1m - 1];
          (*d)[i3][n2d - i2 - 1][n1d - i1 -
                                 1]  += (*m)[0][n2m - 1][n1m - 1];
          (*d)[n3d - i3 - 1][i2][i1] += (*m)[n3m - 1][0][0];
          (*d)[n3d - i3 - 1][n2d - i2 -
                             1][i1] += (*m)[n3m - 1][n2m - 1][0];
          (*d)[n3d - i3 - 1][i2][n1d - i1 -
                                 1] += (*m)[n3m - 1][0][n1m - 1];
          (*d)[n3d - i3 - 1][n2d - i2 - 1][n1d - i1 -
                                           1] +=
            (*m)[n3m - 1][n2m - 1][n1m - 1];
        }
      }
    }
  }

  // top and bottom middle
  for (int i3 = 0; i3 < _padSize3; i3++) {
    for (int i2 = 0; i2 < _padSize2; i2++) {
      for (int i1 = 0; i1 < n1m; i1++) {
        if (_padOption == 0) {
          (*d)[i3][i2][i1 + _padSize1]                     += 0;
          (*d)[i3][n2d - i2 - 1][i1 + _padSize1]           += 0;
          (*d)[n3d - i3 - 1][i2][i1 + _padSize1]           += 0;
          (*d)[n3d - i3 - 1][n2d - i2 - 1][i1 + _padSize1] += 0;
        }
        else {
          (*d)[i3][i2][i1 + _padSize1] += (*m)[0][0][i1];
          (*d)[i3][n2d - i2 - 1][i1 +
                                 _padSize1] += (*m)[0][n2m - 1][i1];
          (*d)[n3d - i3 - 1][i2][i1 +
                                 _padSize1] += (*m)[n3m - 1][0][i1];
          (*d)[n3d - i3 - 1][n2d - i2 - 1][i1 +
                                           _padSize1] +=
            (*m)[n3m - 1][n2m - 1][i1];
        }
      }
    }
  }

  // left and right middle
  for (int i3 = 0; i3 < _padSize3; i3++) {
    for (int i2 = 0; i2 < n2m; i2++) {
      for (int i1 = 0; i1 < _padSize1; i1++) {
        if (_padOption == 0) {
          (*d)[i3][i2 + _padSize2][i1]                     += 0;
          (*d)[i3][i2 + _padSize2][n1d - i1 - 1]           += 0;
          (*d)[n3d - i3 - 1][i2 + _padSize2][i1]           += 0;
          (*d)[n3d - i3 - 1][i2 + _padSize2][n1d - i1 - 1] += 0;
        }
        else {
          (*d)[i3][i2 + _padSize2][i1] += (*m)[0][i2][0];
          (*d)[i3][i2 + _padSize2][n1d - i1 -
                                   1] += (*m)[0][i2][n1m - 1];
          (*d)[n3d - i3 - 1][i2 +
                             _padSize2][i1] += (*m)[n3m - 1][i2][0];
          (*d)[n3d - i3 - 1][i2 + _padSize2][n1d - i1 -
                                             1] += (*m)[n3m - 1][i2][n1m - 1];
        }
      }
    }
  }

  // side middle
  for (int i3 = 0; i3 < n3m; i3++) {
    for (int i2 = 0; i2 < _padSize2; i2++) {
      for (int i1 = 0; i1 < _padSize1; i1++) {
        if (_padOption == 0) {
          (*d)[i3 + _padSize3][i2][i1]                     += 0;
          (*d)[i3 + _padSize3][i2][n1d - i1 - 1]           += 0;
          (*d)[i3 + _padSize3][n2m - i2 - 1][i1]           += 0;
          (*d)[i3 + _padSize3][n2m - i2 - 1][n1d - i1 - 1] += 0;
        }
        else {
          (*d)[i3 + _padSize3][i2][i1] += (*m)[i3][0][0];
          (*d)[i3 + _padSize3][i2][n1d - i1 -
                                   1] += (*m)[i3][0][n1m - 1];
          (*d)[i3 + _padSize3][n2m - i2 -
                               1][i1] += (*m)[i3][n2m - 1][0];
          (*d)[i3 + _padSize3][n2m - i2 - 1][n1d - i1 -
                                             1] += (*m)[i3][n2m - 1][n1m - 1];
        }
      }
    }
  }

  // middle
  #pragma omp for collapse(3)

  for (int i3 = 0; i3 < n3m; i3++) {
    for (int i2 = 0; i2 < n2m; i2++) {
      for (int i1 = 0; i1 < n1m; i1++) {
        (*d)[i3 + _padSize3][i2 + _padSize2][i1 +
                                             _padSize1] += (*m)[i3][i2][i1];
      }
    }
  }
}

// truncates padSize indices from either end of each axis
void PadModel3d::adjoint(const bool                         add,
                         std::shared_ptr<giee::Vector>      model,
                         const std::shared_ptr<giee::Vector>data) {
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  const std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  int n1d =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
  int n2d =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
  int n3d =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;
  int n1m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3m =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

  #pragma omp for collapse(3)

  // // pull data values
  // for (int i3 = 0; i3 < n3m; i3++) {
  //   for (int i2 = 0; i2 < n2m; i2++) {
  //     for (int i1 = 0; i1 < n1m; i1++) {
  //       (*m)[i3][i2][i1] += (*d)[i3 + _padSize][i2 + _padSize][i1 +
  // _padSize];
  //     }
  //   }
  // }
  // corners
  for (int i3 = 0; i3 < _padSize3; i3++) {
    for (int i2 = 0; i2 < _padSize2; i2++) {
      for (int i1 = 0; i1 < _padSize1; i1++) {
        if (_padOption == 0) {}
        else {
          (*m)[0][0][0]       += (*d)[i3][i2][i1];
          (*m)[0][n2m - 1][0] += (*d)[i3][n2d - i2 - 1][i1];
          (*m)[0][0][n1m - 1] += (*d)[i3][i2][n1d - i1 - 1];
          (*m)[0][n2m - 1][n1m -
                           1] +=
            (*d)[i3][n2d - i2 - 1][n1d - i1 - 1];
          (*m)[n3m - 1][0][0] += (*d)[n3d - i3 - 1][i2][i1];
          (*m)[n3m - 1][n2m -
                        1][0] +=
            (*d)[n3d - i3 - 1][n2d - i2 - 1][i1];
          (*m)[n3m - 1][0][n1m -
                           1] +=
            (*d)[n3d - i3 - 1][i2][n1d - i1 - 1];
          (*m)[n3m - 1][n2m - 1][n1m - 1] +=
            (*d)[n3d - i3 - 1][n2d - i2 - 1][n1d - i1 - 1];
        }
      }
    }
  }

  // top and bottom middle
  for (int i3 = 0; i3 < _padSize3; i3++) {
    for (int i2 = 0; i2 < _padSize2; i2++) {
      for (int i1 = 0; i1 < n1m; i1++) {
        if (_padOption == 0) {}
        else {
          (*m)[0][0][i1]       += (*d)[i3][i2][i1 + _padSize1];
          (*m)[0][n2m - 1][i1] += (*d)[i3][n2d - i2 - 1][i1 + _padSize1];
          (*m)[n3m - 1][0][i1] += (*d)[n3d - i3 - 1][i2][i1 + _padSize1];
          (*m)[n3m - 1][n2m - 1][i1]
            += (*d)[n3d - i3 - 1][n2d - i2 - 1][i1 + _padSize1];
        }
      }
    }
  }

  // left and right middle
  for (int i3 = 0; i3 < _padSize3; i3++) {
    for (int i2 = 0; i2 < n2m; i2++) {
      for (int i1 = 0; i1 < _padSize1; i1++) {
        if (_padOption == 0) {}
        else {
          (*m)[0][i2][0]       += (*d)[i3][i2 + _padSize2][i1];
          (*m)[0][i2][n1m - 1] += (*d)[i3][i2 + _padSize2][n1d - i1 - 1];
          (*m)[n3m - 1][i2][0] += (*d)[n3d - i3 - 1][i2 + _padSize2][i1];
          (*m)[n3m - 1][i2][n1m -
                            1] +=
            (*d)[n3d - i3 - 1][i2 + _padSize2][n1d - i1 - 1];
        }
      }
    }
  }

  // side middle
  for (int i3 = 0; i3 < n3m; i3++) {
    for (int i2 = 0; i2 < _padSize2; i2++) {
      for (int i1 = 0; i1 < _padSize1; i1++) {
        if (_padOption == 0) {}
        else {
          (*m)[i3][0][0]       += (*d)[i3 + _padSize3][i2][i1];
          (*m)[i3][0][n1m - 1] += (*d)[i3 + _padSize3][i2][n1d - i1 - 1];
          (*m)[i3][n2m - 1][0] += (*d)[i3 + _padSize3][n2m - i2 - 1][i1];
          (*m)[i3][n2m - 1][n1m -
                            1] +=
            (*d)[i3 + _padSize3][n2m - i2 - 1][n1d - i1 - 1];
        }
      }
    }
  }

  // middle
  #pragma omp for collapse(3)

  for (int i3 = 0; i3 < n3m; i3++) {
    for (int i2 = 0; i2 < n2m; i2++) {
      for (int i1 = 0; i1 < n1m; i1++) {
        (*m)[i3][i2][i1] +=
          (*d)[i3 + _padSize3][i2 + _padSize2][i1 + _padSize1];
      }
    }
  }
}

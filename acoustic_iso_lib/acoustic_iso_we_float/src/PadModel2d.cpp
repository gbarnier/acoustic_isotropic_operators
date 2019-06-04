#include <PadModel2d.h>
using namespace giee;
using namespace waveform;
using namespace SEP;

PadModel2d::PadModel2d(
  const std::shared_ptr<giee::float2DReg>model,
  const std::shared_ptr<giee::float2DReg>data,
  const int                              padSize,
  const int                              padOption)
{
  // model[2][1]

  // both data dimensions should be bigger by 2*padSize
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(
           1).n - 2 * padSize);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(
           2).n - 2 * padSize);

  // spacing should match
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(
           1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(
           2).d);

  // origin of data (ie padded model) axis should be smaller by padSize*d1
  assert(model->getHyper()->getAxis(1).o == data->getHyper()->getAxis(
           1).o + padSize * data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).o == data->getHyper()->getAxis(
           2).o + padSize * data->getHyper()->getAxis(2).d);

  // padOption should be 0 or 1
  assert(padOption == 0 || padOption == 1);

  // set domain and range
  setDomainRange(model, data);

  _padSize   = padSize;
  _padOption = padOption;
}

// adds padSize indices on either end of each axis
void PadModel2d::forward(const bool                         add,
                         const std::shared_ptr<giee::Vector>model,
                         std::shared_ptr<giee::Vector>      data) {
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);

  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  int n1d =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
  int n2d =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;
  int n1m =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2m =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;

  // corners
  for (int i2 = 0; i2 < _padSize; i2++) {
    for (int i1 = 0; i1 < _padSize; i1++) {
      if (_padOption == 0) {
        (*d)[i2][i1]                     += 0;
        (*d)[n2d - i2 - 1][i1]           += 0;
        (*d)[i2][n1d - i1 - 1]           += 0;
        (*d)[n2d - i2 - 1][n1d - i1 - 1] += 0;
      }
      else {
        (*d)[i2][i1]                     += (*m)[0][0];
        (*d)[n2d - i2 - 1][i1]           += (*m)[n2m - 1][0];
        (*d)[i2][n1d - i1 - 1]           += (*m)[0][n1m - 1];
        (*d)[n2d - i2 - 1][n1d - i1 - 1] += (*m)[n2m - 1][n1m - 1];
      }
    }
  }

  // top and bottom middle
  for (int i2 = 0; i2 < _padSize; i2++) {
    for (int i1 = 0; i1 < n1m; i1++) {
      if (_padOption == 0) {
        (*d)[i2][i1 + _padSize]           += 0;
        (*d)[n2d - i2 - 1][i1 + _padSize] += 0;
      }
      else {
        (*d)[i2][i1 + _padSize]           += (*m)[0][i1];
        (*d)[n2d - i2 - 1][i1 + _padSize] += (*m)[n2m - 1][i1];
      }
    }
  }

  // left and right middle
  for (int i2 = 0; i2 < n2m; i2++) {
    for (int i1 = 0; i1 < _padSize; i1++) {
      if (_padOption == 0) {
        (*d)[i2 + _padSize][i1]           += 0;
        (*d)[i2 + _padSize][n1d - i1 - 1] += 0;
      }
      else {
        (*d)[i2 + _padSize][i1]           += (*m)[i2][0];
        (*d)[i2 + _padSize][n1d - i1 - 1] += (*m)[i2][n1m - 1];
      }
    }
  }

  // middle
  #pragma omp for collapse(2)

  for (int i2 = 0; i2 < n2m; i2++) {
    for (int i1 = 0; i1 < n1m; i1++) {
      (*d)[i2 + _padSize][i1 + _padSize] += (*m)[i2][i1];
    }
  }
}

// truncates padSize indices from either end of each axis
void PadModel2d::adjoint(const bool                         add,
                         std::shared_ptr<giee::Vector>      model,
                         const std::shared_ptr<giee::Vector>data) {
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);

  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  int n1d =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
  int n2d =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;
  int n1m =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2m =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;

  // #pragma omp for collapse(2)
  //
  // // pull data values
  // for (int i2 = 0; i2 < n2m; i2++) {
  //   for (int i1 = 0; i1 < n1m; i1++) {
  //     if()
  //     (*m)[i2][i1] += (*d)[i2 + _padSize][i1 + _padSize];
  //   }
  // }
  // corners
  for (int i2 = 0; i2 < _padSize; i2++) {
    for (int i1 = 0; i1 < _padSize; i1++) {
      if (_padOption == 0) {}
      else {
        (*m)[0][0]             += (*d)[i2][i1];
        (*m)[n2m - 1][0]       += (*d)[n2d - i2 - 1][i1];
        (*m)[0][n1m - 1]       += (*d)[i2][n1d - i1 - 1];
        (*m)[n2m - 1][n1m - 1] += (*d)[n2d - i2 - 1][n1d - i1 - 1];
      }
    }
  }

  // top and bottom middle
  for (int i2 = 0; i2 < _padSize; i2++) {
    for (int i1 = 0; i1 < n1m; i1++) {
      if (_padOption == 0) {}
      else {
        (*m)[0][i1]       += (*d)[i2][i1 + _padSize];
        (*m)[n2m - 1][i1] += (*d)[n2d - i2 - 1][i1 + _padSize];
      }
    }
  }

  // left and right middle
  for (int i2 = 0; i2 < n2m; i2++) {
    for (int i1 = 0; i1 < _padSize; i1++) {
      if (_padOption == 0) {}
      else {
        (*m)[i2][0]       += (*d)[i2 + _padSize][i1];
        (*m)[i2][n1m - 1] += (*d)[i2 + _padSize][n1d - i1 - 1];
      }
    }
  }

  // middle
  #pragma omp for collapse(2)

  for (int i2 = 0; i2 < n2m; i2++) {
    for (int i1 = 0; i1 < n1m; i1++) {
      (*m)[i2][i1] += (*d)[i2 + _padSize][i1 + _padSize];
    }
  }
}

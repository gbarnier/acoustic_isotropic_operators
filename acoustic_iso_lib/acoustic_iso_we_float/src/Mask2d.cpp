#include <Mask2d.h>
#include <math.h>
using namespace SEP;

Mask2d::Mask2d(
  const std::shared_ptr<SEP::float2DReg>model,
  const std::shared_ptr<SEP::float2DReg>data,
  int                                    n1min,
  int                                    n1max,
  int                                    n2min,
  int                                    n2max,
  int                                    maskType)
{
  // model and data have the same dimensions
  int n1, n2;

  n1 = model->getHyper()->getAxis(1).n;
  n2 = model->getHyper()->getAxis(2).n;
  assert(n1 == data->getHyper()->getAxis(1).n);
  assert(n2 == data->getHyper()->getAxis(2).n);

  // input mins and maxs are within model and data dimensions
  assert(0 <= n1min);
  assert(n1 >= n1max);
  assert(n1min <= n1max);

  assert(0 <= n2min);
  assert(n2 >= n2max);
  assert(n2min <= n2max);


  // set domain and range
  setDomainRange(model, data);

  _n1min = n1min;
  _n1max = n1max;
  _n2min = n2min;
  _n2max = n2max;

  assert(maskType == 0 || maskType == 1);
  _maskType = maskType;

  _mask.reset(new float2D(boost::extents[n2][n1]));

  double pi = 3.14159;


  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      if (_maskType == 0) {
        if ((i1 < _n1min) || (i1 > _n1max) || (i2 < _n2min) ||
            (i2 > _n2max)) (*_mask)[i2][i1] = 0;
        else (*_mask)[i2][i1] = 1;
      }
      else if (_maskType == 1) {
        (*_mask)[i2][i1] = 1;

        if (i1 < _n1min) {
          (*_mask)[i2][i1] *=
            cos(pi / 2 * (1 - double(i1) / double(_n1min))) *
            cos(pi / 2 *
                (1 - double(i1) / double(_n1min)));
        }

        if (i1 > _n1max) {
          (*_mask)[i2][i1] *=
            cos(pi / 2 * (1 - double(n1 - i1) / double(n1 - _n1max))) * cos(
              pi / 2 * (1 - double(n1 - i1) / double(n1 - _n1max)));
        }

        if (i2 < _n2min) {
          (*_mask)[i2][i1] *=
            cos(pi / 2 * (1 - double(i2) / double(_n2min))) *
            cos(pi / 2 *
                (1 - double(i2) / double(_n2min)));
        }

        if (i2 > _n2max) {
          (*_mask)[i2][i1] *=
            cos(pi / 2 * (1 - double(n2 - i2) / double(n2 - _n2max))) * cos(
              pi / 2 * (1 - double(n2 - i2) / double(n2 - _n2max)));
        }


        // std::cerr << "(*_mask)[" << i3 << "][" << i2 << "][" << i1 << "]="
        // <<
        // (*_mask)[i3][i2][i1] << std::endl;
      }
    }
  }
}

// forward
void Mask2d::forward(const bool                         add,
                     const std::shared_ptr<SEP::float2DReg>model,
                     std::shared_ptr<SEP::float2DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;
   #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*d)[i2][i1] += (*m)[i2][i1] * (*_mask)[i2][i1];
    }
  }
}

// adjoint
void Mask2d::adjoint(const bool                         add,
                     std::shared_ptr<SEP::float2DReg>      model,
                     const std::shared_ptr<SEP::float2DReg>data) const {
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(data))->getHyper()->getAxis(2).n;

#pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*m)[i2][i1] += (*d)[i2][i1] * (*_mask)[i2][i1];
    }
  }
}

Mask2d_complex::Mask2d_complex(
  const std::shared_ptr<SEP::complex2DReg>model,
  const std::shared_ptr<SEP::complex2DReg>data,
  int                                    n1min,
  int                                    n1max,
  int                                    n2min,
  int                                    n2max,
  int                                    maskType)
{
  // model and data have the same dimensions
  _n1 = model->getHyper()->getAxis(1).n;
  _n2 = model->getHyper()->getAxis(2).n;
  assert(_n1 == data->getHyper()->getAxis(1).n);
  assert(_n2 == data->getHyper()->getAxis(2).n);

  // input mins and maxs are within model and data dimensions
  assert(0 <= n1min);
  assert(_n1 >= n1max);
  assert(n1min <= n1max);

  assert(0 <= n2min);
  assert(_n2 >= n2max);
  assert(n2min <= n2max);


  // set domain and range
  setDomainRange(model, data);

  _n1min = n1min;
  _n1max = n1max;
  _n2min = n2min;
  _n2max = n2max;

  assert(maskType == 0 || maskType == 1);
  _maskType = maskType;

  _mask.reset(new complex2D(boost::extents[_n2][_n1]));

  double pi = 3.14159;


  for (int i2 = 0; i2 < _n2; i2++) {
    for (int i1 = 0; i1 < _n1; i1++) {
      if (_maskType == 0) {
        if ((i1 < _n1min) || (i1 > _n1max) || (i2 < _n2min) ||
            (i2 > _n2max)) (*_mask)[i2][i1] = 0;
        else (*_mask)[i2][i1] = 1;
      }
      else if (_maskType == 1) {
        (*_mask)[i2][i1] = 1;

        if (i1 < _n1min) {
          (*_mask)[i2][i1] *=
            cos(pi / 2 * (1 - double(i1) / double(_n1min))) *
            cos(pi / 2 *
                (1 - double(i1) / double(_n1min)));
        }

        if (i1 > _n1max) {
          (*_mask)[i2][i1] *=
            cos(pi / 2 * (1 - double(_n1 - i1) / double(_n1 - _n1max))) * cos(
              pi / 2 * (1 - double(_n1 - i1) / double(_n1 - _n1max)));
        }

        if (i2 < _n2min) {
          (*_mask)[i2][i1] *=
            cos(pi / 2 * (1 - double(i2) / double(_n2min))) *
            cos(pi / 2 *
                (1 - double(i2) / double(_n2min)));
        }

        if (i2 > _n2max) {
          (*_mask)[i2][i1] *=
            cos(pi / 2 * (1 - double(_n2 - i2) / double(_n2 - _n2max))) * cos(
              pi / 2 * (1 - double(_n2 - i2) / double(_n2 - _n2max)));
        }


        // std::cerr << "(*_mask)[" << i3 << "][" << i2 << "][" << i1 << "]="
        // <<
        // (*_mask)[i3][i2][i1] << std::endl;
      }
    }
  }
}

// forward
void Mask2d_complex::forward(const bool                         add,
                     const std::shared_ptr<SEP::complex2DReg>model,
                     std::shared_ptr<SEP::complex2DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  const std::shared_ptr<complex2D> m =
    ((std::dynamic_pointer_cast<complex2DReg>(model))->_mat);
  std::shared_ptr<complex2D> d =
    ((std::dynamic_pointer_cast<complex2DReg>(data))->_mat);

  #pragma omp parallel for collapse(2)
  for (int i2 = 0; i2 < _n2; i2++) {
    for (int i1 = 0; i1 < _n1; i1++) {
      (*d)[i2][i1] += (*m)[i2][i1] * (*_mask)[i2][i1];
    }
  }
}

// adjoint
void Mask2d_complex::adjoint(const bool                         add,
                     std::shared_ptr<SEP::complex2DReg>      model,
                     const std::shared_ptr<SEP::complex2DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<complex2D> m =
    ((std::dynamic_pointer_cast<complex2DReg>(model))->_mat);
  const std::shared_ptr<complex2D> d =
    ((std::dynamic_pointer_cast<complex2DReg>(data))->_mat);

  #pragma omp parallel for collapse(2)
  for (int i2 = 0; i2 < _n2; i2++) {
    for (int i1 = 0; i1 < _n1; i1++) {
      (*m)[i2][i1] += (*d)[i2][i1] * (*_mask)[i2][i1];
    }
  }
}

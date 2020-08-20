#include <Mask4d.h>
#include <math.h>


Mask4d::Mask4d(
  const std::shared_ptr<SEP::float4DReg>model,
  const std::shared_ptr<SEP::float4DReg>data,
  int                                    n1min,
  int                                    n1max,
  int                                    n2min,
  int                                    n2max,
  int                                    n3min,
  int                                    n3max,
  int                                    n4min,
  int                                    n4max,
  int                                    maskType)
{
  // model and data have the same dimensions

  _n1 = model->getHyper()->getAxis(1).n;
  _n2 = model->getHyper()->getAxis(2).n;
  _n3 = model->getHyper()->getAxis(3).n;
  _n4 = model->getHyper()->getAxis(4).n;
  assert(_n1 == data->getHyper()->getAxis(1).n);
  assert(_n2 == data->getHyper()->getAxis(2).n);
  assert(_n3 == data->getHyper()->getAxis(3).n);
  assert(_n4 == data->getHyper()->getAxis(4).n);

  // input mins and maxs are within model and data dimensions
  assert(0 <= n1min);
  assert(_n1 >= n1max);
  assert(n1min <= n1max);

  assert(0 <= n2min);
  assert(_n2 >= n2max);
  assert(n2min <= n2max);

  assert(0 <= n3min);
  assert(_n3 >= n3max);
  assert(n3min <= n3max);

  assert(0 <= n4min);
  assert(_n4 >= n4max);
  assert(n4min <= n4max);

  // set domain and range
  setDomainRange(model, data);

  _n1min = n1min;
  _n1max = n1max;
  _n2min = n2min;
  _n2max = n2max;
  _n3min = n3min;
  _n3max = n3max;
  _n4min = n4min;
  _n4max = n4max;

  assert(maskType == 0 || maskType == 1);
  _maskType = maskType;

  _mask.reset(new float4D(boost::extents[_n4][_n3][_n2][_n1]));

  double pi = 3.14159;
  for (int i4 = 0; i4 < _n4; i4++) {
    for (int i3 = 0; i3 < _n3; i3++) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < _n1; i1++) {
          if (_maskType == 0) {
            if ((i1 < _n1min) || (i1 > _n1max) || (i2 < _n2min) || (i2 > _n2max) ||
                (i3 < _n3min) || (i3 > _n3max) || (i4 < _n4min) || (i4 > _n4max)) (*_mask)[i4][i3][i2][i1] = 0;
            else (*_mask)[i4][i3][i2][i1] = 1;
          }
          else if (_maskType == 1) {
            (*_mask)[i4][i3][i2][i1] = 1;

            if (i1 < _n1min) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(i1) / double(_n1min))) *
                cos(pi / 2 *
                    (1 - double(i1) / double(_n1min)));
            }

            if (i1 > _n1max) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(_n1 - i1) / double(_n1 - _n1max))) * cos(
                  pi / 2 * (1 - double(_n1 - i1) / double(_n1 - _n1max)));
            }

            if (i2 < _n2min) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(i2) / double(_n2min))) *
                cos(pi / 2 *
                    (1 - double(i2) / double(_n2min)));
            }

            if (i2 > _n2max) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(_n2 - i2) / double(_n2 - _n2max))) * cos(
                  pi / 2 * (1 - double(_n2 - i2) / double(_n2 - _n2max)));
            }

            if (i3 < _n3min) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 *
                    (1 - double(i3) / double(_n3min))) *
                cos(pi / 2 *
                    (1 - double(i3) / double(_n3min)));
            }

            if (i3 > _n3max) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(_n3 - i3) / double(_n3 - _n3max))) * cos(
                  pi / 2 * (1 - double(_n3 - i3) / double(_n3 - _n3max)));
            }

            if (i4 < _n4min) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 *
                    (1 - double(i4) / double(_n4min))) *
                cos(pi / 2 *
                    (1 - double(i4) / double(_n4min)));
            }

            if (i3 > _n3max) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(_n4 - i4) / double(_n4 - _n4max))) * cos(
                  pi / 2 * (1 - double(_n4 - i4) / double(_n4 - _n4max)));
            }
            // std::cerr << "(*_mask)[" << i3 << "][" << i2 << "][" << i1 << "]="
            // <<
            // (*_mask)[i3][i2][i1] << std::endl;
          }
        }
      }
    }
  }
}

// forward
void Mask4d::forward(const bool                         add,
                     const std::shared_ptr<SEP::float4DReg>model,
                     std::shared_ptr<SEP::float4DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  const std::shared_ptr<float4D> m =
    ((std::dynamic_pointer_cast<float4DReg>(model))->_mat);
  std::shared_ptr<float4D> d =
    ((std::dynamic_pointer_cast<float4DReg>(data))->_mat);

  #pragma omp parallel for collapse(4)
  for (int i4 = 0; i4 < _n4; i4++) {
    for (int i3 = 0; i3 < _n3; i3++) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < _n1; i1++) {
          (*d)[i4][i3][i2][i1] += (*m)[i4][i3][i2][i1] * (*_mask)[i4][i3][i2][i1];
        }
      }
    }
  }
}

// adjoint
void Mask4d::adjoint(const bool                         add,
                     std::shared_ptr<SEP::float4DReg>      model,
                     const std::shared_ptr<SEP::float4DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float4D> m =
    ((std::dynamic_pointer_cast<float4DReg>(model))->_mat);
  const std::shared_ptr<float4D> d =
    ((std::dynamic_pointer_cast<float4DReg>(data))->_mat);

    #pragma omp parallel for collapse(4)
  for (int i4 = 0; i4 < _n4; i4++) {
    for (int i3 = 0; i3 < _n3; i3++) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < _n1; i1++) {
          (*m)[i4][i3][i2][i1] += (*d)[i4][i3][i2][i1] * (*_mask)[i4][i3][i2][i1];
        }
      }
    }
  }
}


Mask4d_complex::Mask4d_complex(
  const std::shared_ptr<SEP::complex4DReg>model,
  const std::shared_ptr<SEP::complex4DReg>data,
  int                                    n1min,
  int                                    n1max,
  int                                    n2min,
  int                                    n2max,
  int                                    n3min,
  int                                    n3max,
  int                                    n4min,
  int                                    n4max,
  int                                    maskType)
{
  // model and data have the same dimensions

  _n1 = model->getHyper()->getAxis(1).n;
  _n2 = model->getHyper()->getAxis(2).n;
  _n3 = model->getHyper()->getAxis(3).n;
  _n4 = model->getHyper()->getAxis(4).n;
  assert(_n1 == data->getHyper()->getAxis(1).n);
  assert(_n2 == data->getHyper()->getAxis(2).n);
  assert(_n3 == data->getHyper()->getAxis(3).n);
  assert(_n4 == data->getHyper()->getAxis(4).n);

  // input mins and maxs are within model and data dimensions
  assert(0 <= n1min);
  assert(_n1 >= n1max);
  assert(n1min <= n1max);

  assert(0 <= n2min);
  assert(_n2 >= n2max);
  assert(n2min <= n2max);

  assert(0 <= n3min);
  assert(_n3 >= n3max);
  assert(n3min <= n3max);

  assert(0 <= n4min);
  assert(_n4 >= n4max);
  assert(n4min <= n4max);

  // set domain and range
  setDomainRange(model, data);

  _n1min = n1min;
  _n1max = n1max;
  _n2min = n2min;
  _n2max = n2max;
  _n3min = n3min;
  _n3max = n3max;
  _n4min = n4min;
  _n4max = n4max;

  assert(maskType == 0 || maskType == 1);
  _maskType = maskType;

  _mask.reset(new complex4D(boost::extents[_n4][_n3][_n2][_n1]));

  double pi = 3.14159;
  for (int i4 = 0; i4 < _n4; i4++) {
    for (int i3 = 0; i3 < _n3; i3++) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < _n1; i1++) {
          if (_maskType == 0) {
            if ((i1 < _n1min) || (i1 > _n1max) || (i2 < _n2min) || (i2 > _n2max) ||
                (i3 < _n3min) || (i3 > _n3max) || (i4 < _n4min) || (i4 > _n4max)) (*_mask)[i4][i3][i2][i1] = 0;
            else (*_mask)[i4][i3][i2][i1] = 1;
          }
          else if (_maskType == 1) {
            (*_mask)[i4][i3][i2][i1] = 1;

            if (i1 < _n1min) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(i1) / double(_n1min))) *
                cos(pi / 2 *
                    (1 - double(i1) / double(_n1min)));
            }

            if (i1 > _n1max) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(_n1 - i1) / double(_n1 - _n1max))) * cos(
                  pi / 2 * (1 - double(_n1 - i1) / double(_n1 - _n1max)));
            }

            if (i2 < _n2min) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(i2) / double(_n2min))) *
                cos(pi / 2 *
                    (1 - double(i2) / double(_n2min)));
            }

            if (i2 > _n2max) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(_n2 - i2) / double(_n2 - _n2max))) * cos(
                  pi / 2 * (1 - double(_n2 - i2) / double(_n2 - _n2max)));
            }

            if (i3 < _n3min) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 *
                    (1 - double(i3) / double(_n3min))) *
                cos(pi / 2 *
                    (1 - double(i3) / double(_n3min)));
            }

            if (i3 > _n3max) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(_n3 - i3) / double(_n3 - _n3max))) * cos(
                  pi / 2 * (1 - double(_n3 - i3) / double(_n3 - _n3max)));
            }

            if (i4 < _n4min) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 *
                    (1 - double(i4) / double(_n4min))) *
                cos(pi / 2 *
                    (1 - double(i4) / double(_n4min)));
            }

            if (i3 > _n3max) {
              (*_mask)[i4][i3][i2][i1] *=
                cos(pi / 2 * (1 - double(_n4 - i4) / double(_n4 - _n4max))) * cos(
                  pi / 2 * (1 - double(_n4 - i4) / double(_n4 - _n4max)));
            }
            // std::cerr << "(*_mask)[" << i3 << "][" << i2 << "][" << i1 << "]="
            // <<
            // (*_mask)[i3][i2][i1] << std::endl;
          }
        }
      }
    }
  }
}

// forward
void Mask4d_complex::forward(const bool                         add,
                     const std::shared_ptr<SEP::complex4DReg>model,
                     std::shared_ptr<SEP::complex4DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  const std::shared_ptr<complex4D> m =
    ((std::dynamic_pointer_cast<complex4DReg>(model))->_mat);
  std::shared_ptr<complex4D> d =
    ((std::dynamic_pointer_cast<complex4DReg>(data))->_mat);

  #pragma omp parallel for collapse(4)
  for (int i4 = 0; i4 < _n4; i4++) {
    for (int i3 = 0; i3 < _n3; i3++) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < _n1; i1++) {
          (*d)[i4][i3][i2][i1] += (*m)[i4][i3][i2][i1] * (*_mask)[i4][i3][i2][i1];
        }
      }
    }
  }
}

// adjoint
void Mask4d_complex::adjoint(const bool                         add,
                     std::shared_ptr<SEP::complex4DReg>      model,
                     const std::shared_ptr<SEP::complex4DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<complex4D> m =
    ((std::dynamic_pointer_cast<complex4DReg>(model))->_mat);
  const std::shared_ptr<complex4D> d =
    ((std::dynamic_pointer_cast<complex4DReg>(data))->_mat);

    #pragma omp parallel for collapse(4)
  for (int i4 = 0; i4 < _n4; i4++) {
    for (int i3 = 0; i3 < _n3; i3++) {
      for (int i2 = 0; i2 < _n2; i2++) {
        for (int i1 = 0; i1 < _n1; i1++) {
          (*m)[i4][i3][i2][i1] += (*d)[i4][i3][i2][i1] * (*_mask)[i4][i3][i2][i1];
        }
      }
    }
  }
}

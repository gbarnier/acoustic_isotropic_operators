#include <Mask3d.h>
#include <math.h>


Mask3d::Mask3d(
  const std::shared_ptr<SEP::float3DReg>model,
  const std::shared_ptr<SEP::float3DReg>data,
  int                                    n1min,
  int                                    n1max,
  int                                    n2min,
  int                                    n2max,
  int                                    n3min,
  int                                    n3max,
  int                                    maskType)
{
  // model and data have the same dimensions
  int n1, n2, n3;

  n1 = model->getHyper()->getAxis(1).n;
  n2 = model->getHyper()->getAxis(2).n;
  n3 = model->getHyper()->getAxis(3).n;
  assert(n1 == data->getHyper()->getAxis(1).n);
  assert(n2 == data->getHyper()->getAxis(2).n);
  assert(n3 == data->getHyper()->getAxis(3).n);

  // input mins and maxs are within model and data dimensions
  assert(0 <= n1min);
  assert(n1 >= n1max);
  assert(n1min <= n1max);

  assert(0 <= n2min);
  assert(n2 >= n2max);
  assert(n2min <= n2max);

  assert(0 <= n3min);
  assert(n3 >= n3max);
  assert(n3min <= n3max);

  // set domain and range
  setDomainRange(model, data);

  _n1min = n1min;
  _n1max = n1max;
  _n2min = n2min;
  _n2max = n2max;
  _n3min = n3min;
  _n3max = n3max;

  assert(maskType == 0 || maskType == 1);
  _maskType = maskType;

  _mask.reset(new float3D(boost::extents[n3][n2][n1]));

  double pi = 3.14159;

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        if (_maskType == 0) {
          if ((i1 < _n1min) || (i1 > _n1max) || (i2 < _n2min) || (i2 > _n2max) ||
              (i3 < _n3min) || (i3 > _n3max)) (*_mask)[i3][i2][i1] = 0;
          else (*_mask)[i3][i2][i1] = 1;
        }
        else if (_maskType == 1) {
          (*_mask)[i3][i2][i1] = 1;

          if (i1 < _n1min) {
            (*_mask)[i3][i2][i1] *=
              cos(pi / 2 * (1 - double(i1) / double(_n1min))) *
              cos(pi / 2 *
                  (1 - double(i1) / double(_n1min)));
          }

          if (i1 > _n1max) {
            (*_mask)[i3][i2][i1] *=
              cos(pi / 2 * (1 - double(n1 - i1) / double(n1 - _n1max))) * cos(
                pi / 2 * (1 - double(n1 - i1) / double(n1 - _n1max)));
          }

          if (i2 < _n2min) {
            (*_mask)[i3][i2][i1] *=
              cos(pi / 2 * (1 - double(i2) / double(_n2min))) *
              cos(pi / 2 *
                  (1 - double(i2) / double(_n2min)));
          }

          if (i2 > _n2max) {
            (*_mask)[i3][i2][i1] *=
              cos(pi / 2 * (1 - double(n2 - i2) / double(n2 - _n2max))) * cos(
                pi / 2 * (1 - double(n2 - i2) / double(n2 - _n2max)));
          }

          if (i3 < _n3min) {
            (*_mask)[i3][i2][i1] *=
              cos(pi / 2 *
                  (1 - double(i3) / double(_n3min))) *
              cos(pi / 2 *
                  (1 - double(i3) / double(_n3min)));
          }

          if (i3 > _n3max) {
            (*_mask)[i3][i2][i1] *=
              cos(pi / 2 * (1 - double(n3 - i3) / double(n3 - _n3max))) * cos(
                pi / 2 * (1 - double(n3 - i3) / double(n3 - _n3max)));
          }

          // std::cerr << "(*_mask)[" << i3 << "][" << i2 << "][" << i1 << "]="
          // <<
          // (*_mask)[i3][i2][i1] << std::endl;
        }
      }
    }
  }
}

// forward
void Mask3d::forward(const bool                         add,
                     const std::shared_ptr<SEP::float3DReg>model,
                     std::shared_ptr<SEP::float3DReg>      data) const {
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);

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
#pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*d)[i3][i2][i1] += (*m)[i3][i2][i1] * (*_mask)[i3][i2][i1];
      }
    }
  }
}

// adjoint
void Mask3d::adjoint(const bool                         add,
                     std::shared_ptr<SEP::float3DReg>      model,
                     const std::shared_ptr<SEP::float3DReg>data) const{
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);

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
#pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*m)[i3][i2][i1] += (*d)[i3][i2][i1] * (*_mask)[i3][i2][i1];
      }
    }
  }
}

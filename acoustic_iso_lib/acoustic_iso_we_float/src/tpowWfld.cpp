#include <tpowWfld.h>
#include <math.h>
#include <cmath>
using namespace SEP;

tpowWfld::tpowWfld(
  const std::shared_ptr<SEP::float3DReg>model,
  const std::shared_ptr<SEP::float3DReg>data,
  float tpow, float tcenter)
{
  // model and data have the same dimensions
  _n1 = model->getHyper()->getAxis(1).n;
  _n2 = model->getHyper()->getAxis(2).n;
  _n3 = model->getHyper()->getAxis(3).n;
  _d3 = model->getHyper()->getAxis(3).d;
  _o3 = model->getHyper()->getAxis(3).o;
  assert(_n1 == data->getHyper()->getAxis(1).n);
  assert(_n2 == data->getHyper()->getAxis(2).n);
  assert(_n3 == data->getHyper()->getAxis(3).n);

  // set domain and range
  setDomainRange(model, data);

  _tpow=tpow;
  _tcenter=tcenter;
}

// forward
void tpowWfld::forward(const bool                         add,
                     const std::shared_ptr<SEP::float3DReg>model,
                     std::shared_ptr<SEP::float3DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  const std::shared_ptr<float3D> m = model->_mat;
  std::shared_ptr<float3D> d = data->_mat;

   #pragma omp parallel for collapse(3)
  for (int i3 = 0; i3 < _n3; i3++) {
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	float t_dist = std::abs((i3+1)*_d3+_o3 - _tcenter);
        (*d)[i3][i2][i1] += (*m)[i3][i2][i1] * pow(t_dist,_tpow);
      }
    }
  }
}

// adjoint
void tpowWfld::adjoint(const bool                         add,
                     std::shared_ptr<SEP::float3DReg>      model,
                     const std::shared_ptr<SEP::float3DReg>data) const {
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> m = model->_mat;
  const std::shared_ptr<float3D> d = data->_mat;

   #pragma omp parallel for collapse(3)
  for (int i3 = 0; i3 < _n3; i3++) {
    for (int i2 = 0; i2 < _n2; i2++) {
      for (int i1 = 0; i1 < _n1; i1++) {
	float t_dist = std::abs((i3+1)*_d3+_o3 - _tcenter);
        (*m)[i3][i2][i1] += (*d)[i3][i2][i1] * pow(t_dist,_tpow);
      }
    }
  }
}

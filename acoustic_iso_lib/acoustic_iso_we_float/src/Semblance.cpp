#include <Semblance.h>
using namespace giee;
using namespace waveform;
using namespace SEP;

Mask3d::Mask3d(
  const std::shared_ptr<SEP::float3DReg>model,
  const std::shared_ptr<SEP::float3DReg>data)
{
  // model and data have the same 3rd dimensions (same number of cmp gathers)
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);

  // minimum slowness should be positive


  // set domain and range
  setDomainRange(model, data);

  _tau0 = model->getHyper()->getAxis(0).o;
  _t0   = data->getHyper()->getAxis(0).o;
  _s0   = model->getHyper()->getAxis(1).o;
  _off0 = data->getHyper()->getAxis(1).o;
  _dtau = model->getHyper()->getAxis(0).d;
  _dt   = data->getHyper()->getAxis(0).d;
  _ds   = model->getHyper()->getAxis(1).d;
  _doff = data->getHyper()->getAxis(1).d;
  _ntau = model->getHyper()->getAxis(0).n;
  _nt   = data->getHyper()->getAxis(0).n;
  _ns   = model->getHyper()->getAxis(1).n;
  _noff = data->getHyper()->getAxis(1).n;
}

// forward
void Mask3d::forward(const bool                         add,
                     const std::shared_ptr<SEP::Vector>model,
                     std::shared_ptr<SEP::Vector>      data) {
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);

  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  int _nNMO =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;
#pragma omp parallel for collapse(4)

  for (int inmo = 0; inmo < nNMO; i3++) {          // for each NMO
    for (int is = 0; is < _ns; is++) {             // for each slowness value
      for (int itau = 0; itau < _ntau; itau++) {   // for each zero offset time
        for (int ioff = 0; ioff < _noff; ioff++) { // for each offset
          float s   = _s0 + is * _ds;
          float tau = _tau0 + itau * _dtau;
          float off = _off0 + ioff * _doff;
          float t   = sqrt(tau * tau + s * s * off * off);
          int   it  = (_t0 - t) / _dt;
        }
      }
    }
  }
}

// adjoint
void Mask3d::adjoint(const bool                         add,
                     std::shared_ptr<SEP::Vector>      model,
                     const std::shared_ptr<SEP::Vector>data) {
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
#pragma omp parallel for collapse(4)

  for (int inmo = 0; inmo < nNMO; i3++) {          // for each NMO
    for (int is = 0; is < _ns; is++) {             // for each slowness value
      for (int itau = 0; itau < _ntau; itau++) {   // for each zero offset time
        for (int ioff = 0; ioff < _noff; ioff++) { // for each offset
          float s   = _s0 + is * _ds;
          float tau = _tau0 + itau * _dtau;
          float off = _off0 + ioff * _doff;
          float t   = sqrt(tau * tau + s * s * off * off);
          int   it  = (_t0 - t) / _dt;
        }
      }
    }
  }
}

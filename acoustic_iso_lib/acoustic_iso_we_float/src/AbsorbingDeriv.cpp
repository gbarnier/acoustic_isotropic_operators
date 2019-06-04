#include <AbsorbingDeriv.h>
using namespace giee;
using namespace waveform;

AbsorbingDeriv::AbsorbingDeriv(const std::shared_ptr<giee::float2DReg>model,
                               const std::shared_ptr<giee::float2DReg>data,
                               const int                              velPadx,
                               const int                              velPadz)
{
  assert(data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(1).n);
  assert(data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(2).n);
  setDomainRange(model, data);

  _velPadx = velPadx;
  _velPadz = velPadz;
}

void AbsorbingDeriv::forward(const bool                         add,
                             const std::shared_ptr<giee::Vector>model,
                             std::shared_ptr<giee::Vector>      data)
{
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);
  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;
  float d1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).d;
  float d2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).d;


  for (int i1 = 0; i1 < n1; i1++) {
    for (int i2 = 0; i2 < _velPadx; i2++) {
      (*d)[i2][i1] = (*d)[i2][i1] +
                     ((*m)[i2][i1] - (*m)[i2 + 1][i1]) / d2;
    }
  }

  for (int i1 = 0; i1 < n1; i1++) {
    for (int i2 = n2 - _velPadx; i2 < n2; i2++) {
      (*d)[i2][i1] = (*d)[i2][i1] +
                     ((*m)[i2][i1] - (*m)[i2 - 1][i1]) / d2;
    }
  }

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < _velPadz; i1++) {
      (*d)[i2][i1] = (*d)[i2][i1] +
                     ((*m)[i2][i1] - (*m)[i2][i1 + 1]) / d1;
    }
  }

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = n1 - _velPadz; i1 < n1; i1++) {
      (*d)[i2][i1] = (*d)[i2][i1] +
                     ((*m)[i2][i1] - (*m)[i2][i1 - 1]) / d1;
    }
  }
}

void AbsorbingDeriv::adjoint(const bool                         add,
                             const std::shared_ptr<giee::Vector>model,
                             std::shared_ptr<giee::Vector>      data)
{
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);
  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;
  float d1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).d;
  float d2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).d;


  for (int i1 = 0; i1 < n1; i1++) {
    (*m)[0][i1] = (*m)[0][i1] + (*d)[0][i1] / d2;

    for (int i2 = 1; i2 < _velPadx; i2++) {
      (*m)[i2][i1] = (*m)[i2][i1] +
                     ((*d)[i2][i1] - (*d)[i2 - 1][i1]) / d2;
    }
    (*m)[_velPadx][i1] = (*m)[_velPadx][i1] - (*d)[_velPadx - 1][i1] / d2;
  }

  for (int i1 = 0; i1 < n1; i1++) {
    (*m)[n2 - _velPadx - 1][i1] = (*m)[n2 - _velPadx - 1][i1] -
                                  (*d)[n2 - _velPadx][i1] / d2;

    for (int i2 = n2 - _velPadx; i2 < n2 - 1; i2++) {
      (*m)[i2][i1] = (*m)[i2][i1] +
                     ((*d)[i2][i1] - (*d)[i2 + 1][i1]) / d2;
    }
    (*m)[n2 - 1][i1] = (*m)[n2 - 1][i1] + (*d)[n2 - 1][i1] / d2;
  }

  for (int i2 = 0; i2 < n2; i2++) {
    (*m)[i2][0] = (*m)[i2][0] + (*d)[i2][0] / d1;

    for (int i1 = 1; i1 < _velPadz; i1++) {
      (*m)[i2][i1] = (*m)[i2][i1] +
                     ((*d)[i2][i1] - (*d)[i2][i1 - 1]) / d1;
    }
    (*m)[i2][_velPadz] = (*m)[i2][_velPadz] - (*d)[i2][_velPadz - 1] / d1;
  }

  for (int i2 = 0; i2 < n2; i2++) {
    (*m)[i2][n1 - _velPadz - 1] = (*m)[i2][n1 - _velPadz - 1] -
                                  (*d)[i2][n1 - _velPadz] / d1;

    for (int i1 = n1 - _velPadz; i1 < n1 - 1; i1++) {
      (*m)[i2][i1] = (*m)[i2][i1] +
                     ((*d)[i2][i1] - (*d)[i2][i1 + 1]) / d1;
    }
    (*m)[i2][n1 - 1] = (*m)[i2][n1 - 1] + (*d)[i2][n1 - 1] / d1;
  }
}

AbsorbingDeriv_2DCube::AbsorbingDeriv_2DCube(
  const std::shared_ptr<giee::float3DReg>model,
  const std::shared_ptr<giee::float3DReg>data,
  const int                              velPadx,
  const int                              velPadz)
{
  assert(data->getHyper()->sameSize(model->getHyper()));
  setDomainRange(model, data);

  _velPadx = velPadx;
  _velPadz = velPadz;
}

void AbsorbingDeriv_2DCube::forward(const bool                         add,
                                    const std::shared_ptr<giee::Vector>model,
                                    std::shared_ptr<giee::Vector>      data)
{
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);
  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;
  float d2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).d;
  float d3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).d;


  for (int i2 = 0; i2 < n2; i2++) {   // z
    for (int i1 = 0; i1 < n1; i1++) { // t
      (*d)[0][i2][i1] = (*d)[0][i2][i1] - (*m)[0][i2][i1] / d3;
#pragma omp parallel for

      for (int i3 = 1; i3 < _velPadx; i3++) { // x
        (*d)[i3][i2][i1] = (*d)[i3][i2][i1] +
                           ((*m)[i3 - 1][i2][i1] - (*m)[i3][i2][i1]) / d3;
      }
    }
  }

  for (int i2 = 0; i2 < n2; i2++) {   // z
    for (int i1 = 0; i1 < n1; i1++) { // t
      (*d)[n3 - 1][i2][i1] = (*d)[n3 - 1][i2][i1] - (*m)[n3 - 1][i2][i1] / d3;
#pragma omp parallel for

      for (int i3 = n3 - _velPadx; i3 < n3 - 1; i3++) { // x
        (*d)[i3][i2][i1] = (*d)[i3][i2][i1] +
                           ((*m)[i3 + 1][i2][i1] - (*m)[i3][i2][i1]) / d3;
      }
    }
  }


  for (int i3 = 0; i3 < n3; i3++) {   // nx
    for (int i1 = 0; i1 < n1; i1++) { // nt
      (*d)[i3][0][i1] = (*d)[i3][0][i1] - (*m)[i3][0][i1] / d2;
#pragma omp parallel for

      for (int i2 = 1; i2 < _velPadz; i2++) { // nz
        (*d)[i3][i2][i1] = (*d)[i3][i2][i1] +
                           ((*m)[i3][i2 - 1][i1] - (*m)[i3][i2][i1]) / d2;
      }
    }
  }


  for (int i3 = 0; i3 < n3; i3++) {   // nx
    for (int i1 = 0; i1 < n1; i1++) { // nt
      (*d)[i3][n2 - 1][i1] = (*d)[i3][n2 - 1][i1] - (*m)[i3][n2 - 1][i1] / d2;
#pragma omp parallel for

      for (int i2 = n2 - _velPadz; i2 < n2 - 1; i2++) { // nz
        (*d)[i3][i2][i1] = (*d)[i3][i2][i1] +
                           ((*m)[i3][i2 + 1][i1] - (*m)[i3][i2][i1]) / d2;
      }
    }
  }
}

void AbsorbingDeriv_2DCube::adjoint(const bool                         add,
                                    const std::shared_ptr<giee::Vector>model,
                                    std::shared_ptr<giee::Vector>      data)
{
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);
  std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  const std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;
  float d2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).d;
  float d3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).d;


  for (int i2 = 0; i2 < n2; i2++) {   // z
    for (int i1 = 0; i1 < n1; i1++) { // t
      (*m)[_velPadx - 1][i2][i1] = (*m)[_velPadx - 1][i2][i1] -
                                   (*d)[_velPadx - 1][i2][i1] / d3;
#pragma omp parallel for

      for (int i3 = 0; i3 < _velPadx - 1; i3++) { // x
        (*m)[i3][i2][i1] = (*m)[i3][i2][i1] +
                           ((*d)[i3 + 1][i2][i1] - (*d)[i3][i2][i1]) / d3;
      }
    }
  }

  for (int i2 = 0; i2 < n2; i2++) {   // z
    for (int i1 = 0; i1 < n1; i1++) { // t
      (*m)[n3 - _velPadx][i2][i1] = (*m)[n3 - _velPadx][i2][i1] -
                                    (*d)[n3 - _velPadx][i2][i1] / d3;
#pragma omp parallel for

      for (int i3 = n3 - _velPadx + 1; i3 < n3; i3++) { // x
        (*m)[i3][i2][i1] = (*m)[i3][i2][i1] +
                           ((*d)[i3 - 1][i2][i1] - (*d)[i3][i2][i1]) / d3;
      }
    }
  }

  for (int i3 = 0; i3 < n3; i3++) {   // nx
    for (int i1 = 0; i1 < n1; i1++) { // nt
      (*m)[i3][_velPadz - 1][i1] = (*m)[i3][_velPadz - 1][i1] -
                                   (*d)[i3][_velPadz - 1][i1] / d2;
#pragma omp parallel for

      for (int i2 = 0; i2 < _velPadz - 1; i2++) { // nz
        (*m)[i3][i2][i1] = (*m)[i3][i2][i1] +
                           ((*d)[i3][i2 + 1][i1] - (*d)[i3][i2][i1]) / d2;
      }
    }
  }

  for (int i3 = 0; i3 < n3; i3++) {   // nx
    for (int i1 = 0; i1 < n1; i1++) { // nt
      (*m)[i3][n2 - _velPadz][i1] = (*m)[i3][n2 - _velPadz][i1] -
                                    (*d)[i3][n2 - _velPadz][i1] / d2;
#pragma omp parallel for

      for (int i2 = n2 - _velPadz + 1; i2 < n2; i2++) { // nz
        (*m)[i3][i2][i1] = (*m)[i3][i2][i1] +
                           ((*d)[i3][i2 - 1][i1] - (*d)[i3][i2][i1]) / d2;
      }
    }
  }
}

#include <AbsorbingBoundaryConditionV2.h>
using namespace SEP;

AbsorbingBoundaryConditionV2::AbsorbingBoundaryConditionV2(
  const std::shared_ptr<SEP::float3DReg>model,
  const std::shared_ptr<SEP::float3DReg>data,
  const std::shared_ptr<SEP::float2DReg>paddedVel,
  const int                              velPadx,
  const int                              velPadz,
  const float                            absConst,
  const float                            dt)
{
  // assert(data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(1).n);
  // assert(data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(2).n);
  // assert(data->getHyper()->getAxis(1).n ==
  // paddedVel->getHyper()->getAxis(1).n);
  // assert(data->getHyper()->getAxis(2).n ==
  // paddedVel->getHyper()->getAxis(2).n);
  assert(data->getHyper()->sameSize(model->getHyper()));
  assert(data->getHyper()->getAxis(2).n == paddedVel->getHyper()->getAxis(1).n);
  assert(data->getHyper()->getAxis(3).n == paddedVel->getHyper()->getAxis(2).n);
  setDomainRange(model, data);

  std::shared_ptr<SEP::float2DReg> _paddedVel = paddedVel;
  _paddedModelHyper = paddedVel->getHyper();
  _velPadx          = velPadx;
  _velPadz          = velPadz;
  _absConst         = absConst;
  _dt               = dt;

  _w.reset(new float2DReg(_paddedModelHyper));
  _w->scale(0);
  std::shared_ptr<float2D> w_mat =
    ((std::dynamic_pointer_cast<float2DReg>(_w))->_mat);
  std::shared_ptr<float2D> vel =
    ((std::dynamic_pointer_cast<float2DReg>(_paddedVel))->_mat);

  float xScale, zScale, scale, ds, dx, dz, xWeight, zWeight;
  bool  zBorder, xBorder;

  for (int ix = 0; ix < _paddedModelHyper->getAxis(2).n; ix++) {
    for (int iz = 0; iz < _paddedModelHyper->getAxis(1).n; iz++) {
      if (ix < _velPadx) {
        xScale  = float(_velPadx - ix) / float(_velPadx);
        dx      = _paddedModelHyper->getAxis(2).d;
        xWeight = _absConst * xScale * _dt / dx * (*vel)[ix][iz];
        xBorder = 1;
      }
      else if (ix >= _paddedModelHyper->getAxis(2).n - _velPadx) {
        xScale = float(abs(_paddedModelHyper->getAxis(2).n - _velPadx - ix)) /
                 float(_velPadx);
        dx      = _paddedModelHyper->getAxis(2).d;
        xWeight = _absConst * xScale * _dt / dx * (*vel)[ix][iz];
        xBorder = 1;
      }
      else {
        xWeight = 1;
        xBorder = 0;
      }

      if (iz < _velPadz) {
        zScale  = float(abs(iz - _velPadz)) / float(_velPadz);
        dz      = _paddedModelHyper->getAxis(1).d;
        zWeight = _absConst * zScale * _dt / dz * (*vel)[ix][iz];
        zBorder = 1;
      }
      else if (iz >= _paddedModelHyper->getAxis(1).n - _velPadz) {
        zScale = float(abs(_paddedModelHyper->getAxis(1).n - _velPadz - iz)) /
                 float(_velPadz);
        dz      = _paddedModelHyper->getAxis(1).d;
        zWeight = _absConst * zScale * _dt / dz * (*vel)[ix][iz];
        zBorder = 1;
      }
      else {
        zWeight = 1;
        zBorder = 0;
      }

      if (zBorder && xBorder) {
        (*w_mat)[ix][iz] = xWeight * (xScale / (xScale + zScale + .00001)) +
                           zWeight * (zScale / (xScale + zScale + .00001));

        // (*w_mat)[ix][iz] = xWeight*(xScale/(xScale+zScale))+zWeight;
      }
      else if (!zBorder && !xBorder) {
        (*w_mat)[ix][iz] = 0;
      }
      else if (!zBorder) {
        (*w_mat)[ix][iz] = xWeight;
      }
      else if (!xBorder) {
        (*w_mat)[ix][iz] = zWeight;
      }
    }
  }
}

void AbsorbingBoundaryConditionV2::forward(const bool                         add,
                                           const std::shared_ptr<SEP::float3DReg>model,
                                           std::shared_ptr<SEP::float3DReg>      data)
{
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);
  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  const std::shared_ptr<float2D> w_mat =
    ((std::dynamic_pointer_cast<float2DReg>(_w))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

#pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*d)[i3][i2][i1] += (*d)[i3][i2][i1] + (*m)[i3][i2][i1] *
                            (*w_mat)[i3][i2];
      }
    }
  }
}

void AbsorbingBoundaryConditionV2::adjoint(const bool                         add,
                                           const std::shared_ptr<SEP::float3DReg>model,
                                           std::shared_ptr<SEP::float3DReg>      data)
{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);
  std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  const std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  const std::shared_ptr<float2D> w_mat =
    ((std::dynamic_pointer_cast<float2DReg>(_w))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

#pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*m)[i3][i2][i1] += (*m)[i3][i2][i1] + (*d)[i3][i2][i1] *
                            (*w_mat)[i3][i2];
      }
    }
  }
}

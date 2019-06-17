#include <AbsorbingBoundaryCondition.h>
using namespace SEP;

AbsorbingBoundaryCondition::AbsorbingBoundaryCondition(
  const std::shared_ptr<SEP::float2DReg>model,
  const std::shared_ptr<SEP::float2DReg>data,
  const std::shared_ptr<SEP::float2DReg>paddedVel,
  const int                              velPadx,
  const int                              velPadz,
  const float                            absConst,
  const float                            dt)
{
  assert(data->getHyper()->getAxis(1).n == model->getHyper()->getAxis(1).n);
  assert(data->getHyper()->getAxis(2).n == model->getHyper()->getAxis(2).n);
  assert(data->getHyper()->getAxis(1).n == paddedVel->getHyper()->getAxis(1).n);
  assert(data->getHyper()->getAxis(2).n == paddedVel->getHyper()->getAxis(2).n);
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

void AbsorbingBoundaryCondition::forward(const bool                         add,
                                         const std::shared_ptr<SEP::float2DReg>model,
                                         std::shared_ptr<SEP::float2DReg>      data)
{
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);
  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  const std::shared_ptr<float2D> w_mat =
    ((std::dynamic_pointer_cast<float2DReg>(_w))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;


  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*d)[i2][i1] += (*d)[i2][i1] + (*m)[i2][i1] *
                      (*w_mat)[i2][i1];
    }
  }
}

void AbsorbingBoundaryCondition::adjoint(const bool                         add,
                                         const std::shared_ptr<SEP::float2DReg>model,
                                         std::shared_ptr<SEP::float2DReg>      data)
{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);
  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  const std::shared_ptr<float2D> w_mat =
    ((std::dynamic_pointer_cast<float2DReg>(_w))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;


  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*m)[i2][i1] += (*m)[i2][i1] + (*d)[i2][i1] *
                      (*w_mat)[i2][i1];
    }
  }
}

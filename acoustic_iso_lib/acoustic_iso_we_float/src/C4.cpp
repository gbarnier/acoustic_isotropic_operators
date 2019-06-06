#include <C4.h>
using namespace giee;
using namespace waveform;

C4::C4(const std::shared_ptr<SEP::float2DReg>model,
       const std::shared_ptr<SEP::float2DReg>data,
       const std::shared_ptr<SEP::float2DReg>velPadded,
       const int                              velPadx,
       const int                              velPadz,
       const float                            dt
       ) {
  // model and data and velocity domains must match
  assert(data->getHyper()->sameSize(model->getHyper()));
  assert(data->getHyper()->sameSize(velPadded->getHyper()));

  //
  setDomainRange(model, data);
  _velPadded = velPadded;
  _velPadx   = velPadx;
  _velPadz   = velPadz;
  _dt        = dt;
  setWeight();
}

void C4::forward(const bool                         add,
                 const std::shared_ptr<SEP::Vector>model,
                 std::shared_ptr<SEP::Vector>      data)
{
  assert(checkDomainRange(model, data, true));
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;

  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  std::shared_ptr<float2D> w   = _aborbWeight->_mat;
  std::shared_ptr<float2D> vel = _velPadded->_mat;

  if (!add) data->scale(0.);

  // loop through data
  #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      // data = (w-I)*v*v*dt*dt*model
      // data = w*v*v*dt*dt*model - v*v*dt*dt*model
      float c = (*vel)[i2][i1] * (*vel)[i2][i1] * _dt * _dt;
      (*d)[i2][i1] += c * (*w)[i2][i1] * (*m)[i2][i1] - c *
                      (*m)[i2][i1];
    }
  }
}

void C4::adjoint(const bool                         add,
                 std::shared_ptr<SEP::Vector>      model,
                 const std::shared_ptr<SEP::Vector>data)
{
  assert(checkDomainRange(model, data, true));
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;

  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  std::shared_ptr<float2D> w   = _aborbWeight->_mat;
  std::shared_ptr<float2D> vel = _velPadded->_mat;

  if (!add) model->scale(0.);

  // loop through model
  #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      // model = (w-I)*v*v*dt*dt*data
      // model = w*v*v*dt*dt*data - v*v*dt*dt*data
      float c = (*vel)[i2][i1] * (*vel)[i2][i1] * _dt * _dt;
      (*m)[i2][i1] += c * (*w)[i2][i1] * (*d)[i2][i1] - c *
                      (*d)[i2][i1];
    }
  }
}

void C4::setWeight() {
  std::shared_ptr<SEP::hypercube> _paddedModelHyper = _velPadded->getHyper();
  _aborbWeight.reset(new float2DReg(_paddedModelHyper));
  _aborbWeight->scale(0);
  std::shared_ptr<float2D> w =
    ((std::dynamic_pointer_cast<float2DReg>(_aborbWeight))->_mat);
  std::shared_ptr<float2D> vel =
    ((std::dynamic_pointer_cast<float2DReg>(_velPadded))->_mat);

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
        (*w)[ix][iz] = xWeight * (xScale / (xScale + zScale + .00001)) +
                       zWeight * (zScale / (xScale + zScale + .00001));

        // (*w)[ix][iz] = xWeight*(xScale/(xScale+zScale))+zWeight;
      }
      else if (!zBorder && !xBorder) {
        (*w)[ix][iz] = 0;
      }
      else if (!zBorder) {
        (*w)[ix][iz] = xWeight;
      }
      else if (!xBorder) {
        (*w)[ix][iz] = zWeight;
      }
    }
  }
}

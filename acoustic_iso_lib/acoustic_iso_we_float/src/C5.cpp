#include <C5.h>
using namespace giee;
using namespace waveform;

C5::C5(
  const std::shared_ptr<giee::float2DReg>model,
  const std::shared_ptr<giee::float2DReg>data,
  const std::shared_ptr<giee::float2DReg>velPadded,
  const int                              velPadx,
  const int                              velPadz,
  const float                            dt
  ) {
  // model and data and velocity domains must match
  assert(data->getHyper()->sameSize(model->getHyper()));
  assert(data->getHyper()->sameSize(velPadded->getHyper()));


  setDomainRange(model, data);
  _velPadded = velPadded;
  _velPadx   = velPadx;
  _velPadz   = velPadz;
  _temp0.reset(new float2DReg(model->getHyper()));
  _temp1.reset(new float2DReg(model->getHyper()));
  _dt = dt;
  _C2.reset(new waveform::C2(model,
                             data,
                             _velPadded,
                             _dt));
  _G.reset(new waveform::G(model, data, _velPadx, _velPadz));
  setWeight();
}

void C5::forward(const bool                         add,
                 const std::shared_ptr<giee::Vector>model,
                 std::shared_ptr<giee::Vector>      data)
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
  std::shared_ptr<float2D> t0  = _temp0->_mat;
  std::shared_ptr<float2D> vel = _velPadded->_mat;

  if (!add) data->scale(0.);

  // temp0 = v*v*dt*dt*lapl(model)+2*model
  _C2->forward(0, model, _temp0);

  // loop through data
  #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      // data = (I-w)(v*v*dt*dt*lapl(model)+2*model)
      // data = (I-w)*temp0
      // data = temp0 - w*temp0
      (*d)[i2][i1] += double((*t0)[i2][i1]) - double((*w)[i2][i1]) *
                      double((*t0)[i2][i1]);
    }
  }

  // temp0= (d/ds)(model)
  _G->forward(0, model, _temp0);

  // loop through data
    #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      // data = (I-w)(v*v*dt*dt*lapl(model)+2*model) + w(I-v*dt*(d/ds))(model)
      // data = (I-w)(v*v*dt*dt*lapl(model)+2*model) +
      //  w*model-w*v*dt*(d/ds)(model)
      // data = data + w*model-w*v*dt*temp0
      (*d)[i2][i1] += double((*w)[i2][i1]) * double((*m)[i2][i1]) -
                      double((*w)[i2][i1]) * double((*vel)[i2][i1]) * _dt *
                      double((*t0)[i2][i1]);
    }
  }
}

void C5::adjoint(const bool                         add,
                 std::shared_ptr<giee::Vector>      model,
                 const std::shared_ptr<giee::Vector>data)
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
  std::shared_ptr<float2D> t0  = _temp0->_mat;
  std::shared_ptr<float2D> t1  = _temp1->_mat;
  std::shared_ptr<float2D> vel = _velPadded->_mat;

  if (!add) model->scale(0.);

  // temp0 = (I-w)data
  // temp0 = data - w*data
  #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*t0)[i2][i1] = double((*d)[i2][i1]) - double((*w)[i2][i1]) *
                      double((*d)[i2][i1]);
    }
  }

  _C2->adjoint(add, model, _temp0); // model = (lapl'*v*v*dt*dt+2I)(I-w)(data)

  // temp0 = v*dt*w*data
  #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*t0)[i2][i1] = double((*d)[i2][i1]) *
                      double((*w)[i2][i1]) *
                      double((*vel)[i2][i1]) * _dt;
    }
  }

  _G->adjoint(0, _temp1, _temp0); // temp1= (d/ds)'(v*dt*w*data)

  #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*m)[i2][i1] += double((*w)[i2][i1]) * double((*d)[i2][i1]) -
                      double((*t1)[i2][i1]); // -
      // (*w)[i2][i1]
      // *(*vel)[i2][i1] * _dt *
      //  (*t0)[i2][i1];
    }
  }
}

void C5::setWeight() {
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

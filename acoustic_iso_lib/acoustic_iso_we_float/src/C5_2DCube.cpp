#include <C5_2DCube.h>
using namespace giee;
using namespace waveform;

C5_2DCube::C5_2DCube(
  const std::shared_ptr<giee::float3DReg>model,
  const std::shared_ptr<giee::float3DReg>data,
  const std::shared_ptr<giee::float2DReg>velPadded,
  const int                              velPadx,
  const int                              velPadz,
  const float                            dt
  ) {
  // model and data and velocity domains must match
  assert(data->getHyper()->sameSize(model->getHyper()));
  assert(model->getHyper()->getAxis(2).n == velPadded->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(3).n == velPadded->getHyper()->getAxis(2).n);


  setDomainRange(model, data);
  _velPadded = velPadded;
  _velPadx   = velPadx;
  _velPadz   = velPadz;
  _temp0.reset(new float3DReg(model->getHyper()));
  _temp1.reset(new float3DReg(model->getHyper()));
  _dt = dt;
  _C2.reset(new waveform::C2_2DCube(model,
                                    data,
                                    _velPadded,
                                    _dt));
  _G.reset(new waveform::G_2DCube(model, data, _velPadx, _velPadz));
  setWeight();
}

void C5_2DCube::forward(const bool                         add,
                        const std::shared_ptr<giee::Vector>model,
                        std::shared_ptr<giee::Vector>      data)
{
  assert(checkDomainRange(model, data, true));
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);

  // std::shared_ptr<float2D> w   = _aborbWeight->_mat;
  // std::shared_ptr<float3D> t0  = _temp0->_mat;
  // std::shared_ptr<float2D> vel = _velPadded->_mat;

  if (!add) data->scale(0.);

  // temp0 = v*v*dt*dt*lapl(model)+2*model
  _C2->forward(0, model, _temp0);

  // loop through data
  #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        // data = (I-w)(v*v*dt*dt*lapl(model)+2*model)
        // data = (I-w)*temp0
        // data = temp0 - w*temp0
        (*d)[i3][i2][i1] += (*_temp0->_mat)[i3][i2][i1] -
                            (*_absorbWeight->_mat)[i3][i2] *
                            (*_temp0->_mat)[i3][i2][i1];
      }
    }
  }

  // temp0= (d/ds)(model)
  _G->forward(0, model, _temp0);

  // loop through data
    #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        // data = (I-w)(v*v*dt*dt*lapl(model)+2*model) + w(I-v*dt*(d/ds))(model)
        // data = (I-w)(v*v*dt*dt*lapl(model)+2*model) +
        //  w*model-w*v*dt*(d/ds)(model)
        // data = data + w*model-w*v*dt*temp0
        (*d)[i3][i2][i1] += (*_absorbWeight->_mat)[i3][i2] * (*m)[i3][i2][i1] -
                            (*_absorbWeight->_mat)[i3][i2] *
                            (*_velPadded->_mat)[i3][i2] * _dt *
                            (*_temp0->_mat)[i3][i2][i1];
      }
    }
  }
}

void C5_2DCube::adjoint(const bool                         add,
                        std::shared_ptr<giee::Vector>      model,
                        const std::shared_ptr<giee::Vector>data)
{
  assert(checkDomainRange(model, data, true));
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

  std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  const std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);

  // std::shared_ptr<float2D> w   = _absorbWeight->_mat;
  // std::shared_ptr<float3D> t0  = _temp0->_mat;
  // std::shared_ptr<float2D> vel = _velPadded->_mat;

  if (!add) model->scale(0.);

  // temp0 = (I-w)data
  // temp0 = data - w*data
  #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*_temp0->_mat)[i3][i2][i1] = (*d)[i3][i2][i1] -
                                      (*_absorbWeight->_mat)[i3][i2] *
                                      (*d)[i3][i2][i1];
      }
    }
  }
  _C2->adjoint(add, model, _temp0); // model = (lapl'*v*v*dt*dt+2I)(I-w)(data)
  #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*_temp0->_mat)[i3][i2][i1] = (*d)[i3][i2][i1]
                                      * (*_absorbWeight->_mat)[i3][i2]
                                      * (*_velPadded->_mat)[i3][i2] * _dt;
      }
    }
  }

  _G->adjoint(0, _temp1, _temp0); // temp0= (d/ds)'(data)

#pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*m)[i3][i2][i1] += (*_absorbWeight->_mat)[i3][i2] * (*d)[i3][i2][i1] -
                            (*_temp1->_mat)[i3][i2][i1];
      }
    }
  }
}

void C5_2DCube::setWeight() {
  std::shared_ptr<SEP::hypercube> _paddedModelHyper = _velPadded->getHyper();
  _absorbWeight.reset(new float2DReg(_paddedModelHyper));
  _absorbWeight->scale(0);
  std::shared_ptr<float2D> w =
    ((std::dynamic_pointer_cast<float2DReg>(_absorbWeight))->_mat);
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

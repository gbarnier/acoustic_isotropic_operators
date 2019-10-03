#include <AcousticProp.h>
#include <C4_2DCube.h>
using namespace giee;
using namespace waveform;
using namespace SEP;

AcousticProp::AcousticProp(
  const std::shared_ptr<SEP::float3DReg>model,
  const std::shared_ptr<SEP::float3DReg>data,
  const std::shared_ptr<SEP::float2DReg>velPadded)
{
  // model and data and source have same dimensions
  assert(data->getHyper()->sameSize(model->getHyper()));

  // assert(data->getHyper()->sameSize(source->getHyper()));

  // model (and data) axis 2 and 3 match velPadded axis 2 and 3

  assert(model->getHyper()->getAxis(2).n == velPadded->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(3).n == velPadded->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(2).d == velPadded->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(3).d == velPadded->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(2).o == velPadded->getHyper()->getAxis(1).o);
  assert(model->getHyper()->getAxis(3).o == velPadded->getHyper()->getAxis(2).o);

  _dt = model->getHyper()->getAxis(1).d;
  _nt = model->getHyper()->getAxis(1).n;

  _temp0.reset(new float2DReg(model->getHyper()->getAxis(2).n,
                              model->getHyper()->getAxis(
                                3).n));
  _temp1.reset(new float2DReg(model->getHyper()->getAxis(2).n,
                              model->getHyper()->getAxis(
                                3).n));

  _temp0.reset(new float2DReg(velPadded->getHyper()));
  _temp1.reset(new float2DReg(velPadded->getHyper()));
  _temp2.reset(new float2DReg(velPadded->getHyper()));


  // initialize C5
  _C2.reset(new waveform::C2(
              _temp0,
              _temp1,
              velPadded,
              _dt));

  // set domain and range
  setDomainRange(model, data);
}

// forward
void AcousticProp::forward(const bool                         add,
                           const std::shared_ptr<SEP::Vector>model,
                           std::shared_ptr<SEP::Vector>      data) {
  assert(checkDomainRange(model, data));

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

  // _C4f.reset(new
  // float3DReg((std::dynamic_pointer_cast<float3DReg>(data))->getHyper()));
  // _C4->forward(0, model, _C4f);
  // _C4f->scaleAdd(0, model, 1);

  _temp0->scale(0.);
  _temp1->scale(0.);
  _temp2->scale(0.);

  // data[t=0] = 0
  // temp0=data[t=0]
  #pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      // (*d)[i3][i2][0] += 0;
      (*_temp0->_mat)[i3][i2] += (*d)[i3][i2][0];
    }
  }

  // temp1=C2data[0]
  // temp1=C2temp0
  _C2->forward(0, _temp0, _temp1);

  // data[1] = C5data[0] + model[0]
  // data[1]=temp1 + model[0]
  #pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      (*_temp1->_mat)[i3][i2] += (*m)[i3][i2][0];
      (*d)[i3][i2][1]         += (*_temp1->_mat)[i3][i2];
    }
  }

  for (int it = 2; it < _nt; it++) {
    // temp2 = C2data[it-1]
    // temp2=C2temp1
    _C2->forward(0, _temp1, _temp2);

    // _C6->forward(1, _temp0, _temp2);

    // data[it] = C5data[it-1] + model[it-1]
    // data[it]=temp2 + model[it-1]
    #pragma omp parallel for collapse(2)

    for (int i3 = 0; i3 < n3; i3++) {
      for (int i2 = 0; i2 < n2; i2++) {
        (*_temp2->_mat)[i3][i2] += double((*m)[i3][i2][it - 1]) -
                                   double((*_temp0->_mat)[i3][i2]);
        (*d)[i3][i2][it] += (*_temp2->_mat)[i3][i2];
      }
    }

    swap(_temp0, _temp1);

    // swap(t0,     t1);
    swap(_temp1, _temp2);

    // swap(t1,     t2);
  }
}

// adjoint
void AcousticProp::adjoint(const bool                         add,
                           std::shared_ptr<SEP::Vector>      model,
                           const std::shared_ptr<SEP::Vector>data) {
  assert(checkDomainRange(model, data));

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


  // _C4f.reset(new
  // float3DReg((std::dynamic_pointer_cast<float3DReg>(model))->getHyper()));
  // _C4->adjoint(0, _C4f, data);
  // _C4f->scaleAdd(0, data, 1);

  // // t=0
  // // temp0 = 0;
  _temp0->scale(0.);
  _temp1->scale(0.);
  _temp2->scale(0.);

  // model[nt-1] = 0
  // temp0=data[nt-1]
  #pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      // (*m)[i3][i2][_nt - 1] += 0;
      (*_temp0->_mat)[i3][i2] += (*m)[i3][i2][_nt - 1];
    }
  }

  // temp1=C5'model[nt-1]
  // temp1=C5'temp0
  _C2->adjoint(0, _temp1, _temp0);

  // model[nt-2] = C5'model[nt-1] + data[nt-1]
  // model[nt-2]=temp1 + data[nt-1]
  #pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      (*_temp1->_mat)[i3][i2] += (*d)[i3][i2][_nt - 1];
      (*m)[i3][i2][_nt - 2]   += (*_temp1->_mat)[i3][i2];
    }
  }

  for (int it = _nt - 3; it >= 0; it--) {
    // temp2 = C5'model[it+1]
    // temp2=C5'temp1
    _C2->adjoint(0, _temp2, _temp1);

    // _C6->adjoint(1, _temp2, _temp0);

    // model[it] = C5'model[it+1] + data[it+1]
    // model[it]=temp2 + data[it+1]
    #pragma omp parallel for collapse(2)

    for (int i3 = 0; i3 < n3; i3++) {
      for (int i2 = 0; i2 < n2; i2++) {
        (*_temp2->_mat)[i3][i2] += double((*d)[i3][i2][it + 1]) -
                                   double((*_temp0->_mat)[i3][i2]);
        (*m)[i3][i2][it] += (*_temp2->_mat)[i3][i2];
      }
    }

    swap(_temp0, _temp1);

    // swap(t0,     t1);
    swap(_temp1, _temp2);

    // swap(t1,     t2);
  }
}

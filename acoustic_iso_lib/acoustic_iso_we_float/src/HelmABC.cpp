#include <HelmABC.h>
using namespace giee;
using namespace waveform;
using namespace SEP;

HelmABC::HelmABC(
  const std::shared_ptr<giee::float3DReg>model,
  const std::shared_ptr<giee::float3DReg>data,
  const std::shared_ptr<giee::float2DReg>velPadded,
  const int                              velPadx,
  const int                              velPadz)
{
  // model and data have same dimensions
  assert(data->getHyper()->sameSize(model->getHyper()));

  // model (and data) axis 2 and 3 match velPadded axis 2 and 3

  assert(model->getHyper()->getAxis(2).n == velPadded->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(3).n == velPadded->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(2).d == velPadded->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(3).d == velPadded->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(2).o == velPadded->getHyper()->getAxis(1).o);
  assert(model->getHyper()->getAxis(3).o == velPadded->getHyper()->getAxis(2).o);

  _dt = model->getHyper()->getAxis(1).d;

  // initialize C4 op
  _C4R.reset(new waveform::C4R_2DCube(model,
                                      data,
                                      velPadded,
                                      velPadx,
                                      velPadz,
                                      _dt));

  // _C4f = (w-I)*v*v*dt*dt*source
  // _C4f.reset(new float3DReg(model->getHyper()));

  // _C4->forward(0, model, _C4f);

  // initialize C6 op
  _C6.reset(new waveform::C6_2DCube(model,
                                    data,
                                    velPadded,
                                    velPadx,
                                    velPadz,
                                    _dt));

  // initialize C5
  _C5.reset(new waveform::C5_2DCube(
              model,
              data,
              velPadded,
              velPadx,
              velPadz,
              _dt));

  _temp0.reset(new float3DReg(model->getHyper()));
  _temp1.reset(new float3DReg(model->getHyper()));
  _temp2.reset(new float3DReg(model->getHyper()));

  // set domain and range
  setDomainRange(model, data);
}

// forward
void HelmABC::forward(const bool                         add,
                      const std::shared_ptr<giee::Vector>model,
                      std::shared_ptr<giee::Vector>      data) {
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);

  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);

  // std::shared_ptr<float3D> c4 =
  //   ((std::dynamic_pointer_cast<float3DReg>(_C4f))->_mat);
  std::shared_ptr<float3D> t0 =
    ((std::dynamic_pointer_cast<float3DReg>(_temp0))->_mat);
  std::shared_ptr<float3D> t1 =
    ((std::dynamic_pointer_cast<float3DReg>(_temp1))->_mat);
  std::shared_ptr<float3D> t2 =
    ((std::dynamic_pointer_cast<float3DReg>(_temp2))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;


  _temp0->scale(0.);
  _temp1->scale(0.);
  _temp2->scale(0.);

  // data[it] = 1/C4(-model[it+1] + C5(model)[it] + C6(model)[it-2])
  // _C4->forward(0, model, _C4f);

  // // temp0[it] += -model[it+1]
  // #pragma omp parallel for collapse(3)
  //
  // for (int i3 = 0; i3 < n3; i3++) {
  //   for (int i2 = 0; i2 < n2; i2++) {
  //     for (int i1 = 0; i1 < n1 - 1; i1++) {
  //       (*t0)[i3][i2][i1] = -1 * (*m)[i3][i2][i1 + 1];
  //     }
  //   }
  // }


  // temp1 += _C5model[it]
  _C5->forward(0, model, _temp1);

  // temp2 = _C6(model)
  _C6->forward(0, model, _temp2);

  // temp0[it] += -model[it+1] + _C5[it] + _C6[it-1]
  #pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      (*t0)[i3][i2][0] = -1 * (*m)[i3][i2][0];
      (*t0)[i3][i2][1] = (*t1)[i3][i2][0] - (*m)[i3][i2][1];

      // (*t0)[i3][i2][n1 - 1] = (*t2)[i3][i2][n1 - 2] + (*t1)[i3][i2][n1 - 1];
    }
  }


  // temp0[it] += -model[it+1] + _C5[it] + _C6[it-1]
  #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 2; i1 < n1; i1++) {
        (*t0)[i3][i2][i1] = (*t2)[i3][i2][i1 - 2] + (*t1)[i3][i2][i1 - 1] -
                            (*m)[i3][i2][i1];
      }
    }
  }

  _C4R->forward(1, _temp0, data);

  // data->scale(-1);
}

// adjoint
void HelmABC::adjoint(const bool                         add,
                      std::shared_ptr<giee::Vector>      model,
                      const std::shared_ptr<giee::Vector>data) {
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  const std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);

  //  std::shared_ptr<float3D> c4 =
  //    ((std::dynamic_pointer_cast<float3DReg>(_C4f))->_mat);
  std::shared_ptr<float3D> t0 =
    ((std::dynamic_pointer_cast<float3DReg>(_temp0))->_mat);
  std::shared_ptr<float3D> t1 =
    ((std::dynamic_pointer_cast<float3DReg>(_temp1))->_mat);
  std::shared_ptr<float3D> t2 =
    ((std::dynamic_pointer_cast<float3DReg>(_temp2))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;

  _temp0->scale(0.);
  _temp1->scale(0.);
  _temp2->scale(0.);

  // model[it] = -data[it-1] + C5(data)[it] + C6(data)[it+1]
  // temp0 = C4'(data)
  _C4R->adjoint(0, _temp0, data);

  // _temp0->scaleAdd(0, data, 1);

  // model = -C4'(data[it-1]) + _C5'(C4'(data[it]))
  _C5->adjoint(0, _temp1, _temp0);

  // temp1 = _C6'(data)
  _C6->adjoint(0, _temp2, _temp0);

  #pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      (*m)[i3][i2][n1 - 1] += -1 * (*t0)[i3][i2][n1 - 1];
      (*m)[i3][i2][n1 - 2] += -1 * (*t0)[i3][i2][n1 - 2] + (*t1)[i3][i2][n1 - 1];
    }
  }

  // model = -C4'(data[it-1]) + C5'(C4'(data[it])) + C6'(C4'(data[it+1]))
  #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1 - 2; i1++) {
        (*m)[i3][i2][i1] += -1 * (*t0)[i3][i2][i1] +
                            (*t1)[i3][i2][i1 + 1] + (*t2)[i3][i2][i1 + 2];
      }
    }
  }

  // model->scale(-1);
}

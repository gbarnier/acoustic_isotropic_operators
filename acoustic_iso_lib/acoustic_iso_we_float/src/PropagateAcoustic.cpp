#include <PropogateAcoustic.h>
using namespace giee;
using namespace waveform;

PropogateAcoustic::PropogateAcoustic(const std::shared_ptr<giee::float3DReg>model,
                                     const std::shared_ptr<giee::float3DReg>data,
                                     const waveform::PropogateStepper       StepperOp,
                                     const waveform::BoundaryCondition      BoundaryOp)
{
  // check that domain of pressure before and after prop are the same
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);

  // check if StepperOp has same domain and range
  assert(StepperOp->checkDomainRange(model, data));

  // ensure absorbing boundary dimensions math that of slices of data/model
  assert(BoundaryOp->getDomain()->getHyper()->getAxis(
           1) == data->getHyper()->getAxis(2).n);
  assert(BoundaryOp->getDomain()->getHyper()->getAxis(
           2) == data->getHyper()->getAxis(3).n);

  // good to go
  setDomainRange(model, data);
  _StepperOp  = StepperOp;
  _BoundaryOp = BoundaryOp;
}

void PropogateAcoustic::forward(const bool                         add,
                                const std::shared_ptr<giee::Vector>f,
                                std::shared_ptr<giee::Vector>      p) {
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);

  std::shared_ptr<float3D> f =
    ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);
  std::shared_ptr<float3D> p =
    ((std::dynamic_pointer_cast<float3DReg>(buffer))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;

  ////scale entire model 'source' cube. C4 equivalent
  // f = (w-I)*(v^2)*(dt^2)f
  std::shared_ptr<float2DReg> ftemp = f->clone();
  _ScaleSourceOp->forward(0, f, ftemp); // ftemp = (v^2)*(dt^2)f
  _BoundaryOp->forward(0, ftemp, f);    // f = w * ftemp
  f->scaleAdd(1, ftemp, -1);            // f = f - ftemp

  // Prop

  for (int i1 = 2; i1 < n1; i1++) {
    // grab one time slice in a float2DReg

    float2DReg pNew, pCur, pOld, fCur;


    // PropogateStepper fwd.
    _StepperOp->forward(add, model, data, it);
  }
}

void PropogateAcoustic::adjoint(const bool                         add,
                                std::shared_ptr<giee::Vector>      model,
                                const std::shared_ptr<giee::Vector>data) {}

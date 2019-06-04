#include <PropagateStepperAcoustic.h>
using namespace giee;
using namespace waveform;

PropagateStepperAcoustic::PropagateStepperAcoustic(
  const std::shared_ptr<giee::float2DReg>           model,
  const std::shared_ptr<giee::float2DReg>           data,
  const std::shared_ptr<giee::float2DReg>           velPadded,
  const std::shared_ptr<giee::float3DReg>           sourceCube,
  const std::shared_ptr<waveform::BoundaryCondition>BC,
  const int                                         velPadx,
  const int                                         velPadz
  ) {
  // model and data and velocity domains must match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(1).n == velPadded->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(1).n == sourceCube->getHyper()->getAxis(2).n);

  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(2).n == velPadded->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(2).n == sourceCube->getHyper()->getAxis(3).n);

  // Boundary Condition Operator domain must match model and data
  assert(BC->checkDomainRange(model, data));

  // Boundaries match set domain and range and member variables
  setDomainRange(model, data);
  _sourceCube = sourceCube;
  int dt = _sourceCube->getHyper()->getAxis(1).d;
  _it = 2;

  // Initialize operators
  _C4.reset(new waveform::C4(model,
                             data,
                             velPadded,
                             BC,
                             dt));
  _C5.reset(new waveform::C5(model,
                             data,
                             velPadded,
                             BC,
                             velPadx,
                             velPadz,
                             dt));
  _C6.reset(new waveform::C6(model,
                             data,
                             BC));
}

void PropagateStepperAcoustic::forward(const bool                         add,
                                       const std::shared_ptr<giee::Vector>pCur,
                                       std::shared_ptr<giee::Vector>      pNew)
{
  std::shared_ptr<giee::float2DReg> temp0;

  if (!add) pNew->scale(0.);
  else temp0 = std::dynamic_pointer_cast<float2DReg>(pNew->clone());

  // grab source slice at _it
  int n2                                 = _sourceCur->getHyper()->getAxis(1).n;
  int n3                                 = _sourceCur->getHyper()->getAxis(2).n;
  std::shared_ptr<float2D> _sourceCurMat =
    ((std::dynamic_pointer_cast<float2DReg>(_sourceCur))->_mat);
  std::shared_ptr<float3D> _sourceCubeMat =
    ((std::dynamic_pointer_cast<float3DReg>(_sourceCube))->_mat);

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      (*_sourceCurMat)[i3][i2] = (*_sourceCubeMat)[i3][i2][_it];
    }
  }

  // pNew = C4(_sourceCur)
  _C4->forward(0, _sourceCur, pNew);

  // pNew =  C4(_sourceCur) + C5(pCur)
  _C5->forward(1, pCur, pNew);

  // pNew = C4(_sourceCur) + C5(pCur) + C6(_pOld);
  _C6->forward(1, _pOld, pNew);

  // increment time
  _it++;

  // pOld=pCur
  _pOld->scaleAdd(0, pCur, 1);

  if (add) pNew->scaleAdd(1, temp0, 1);
}

void PropagateStepperAcoustic::adjoint(const bool                         add,
                                       std::shared_ptr<giee::Vector>      pCur,
                                       const std::shared_ptr<giee::Vector>pNew)
{
  std::shared_ptr<giee::float2DReg> temp0;

  if (!add) pCur->scale(0.);
  else temp0 = std::dynamic_pointer_cast<float2DReg>(pCur->clone());


  if (add) pCur->scaleAdd(1, temp0, 1);
}

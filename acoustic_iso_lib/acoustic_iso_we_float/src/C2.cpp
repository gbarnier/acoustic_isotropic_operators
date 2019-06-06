#include <C2.h>
using namespace giee;
using namespace waveform;

C2::C2(
  const std::shared_ptr<SEP::float2DReg>model,
  const std::shared_ptr<SEP::float2DReg>data,
  const std::shared_ptr<SEP::float2DReg>velPadded,
  const float                            dt
  ) {
  // model and data and velocity domains must match
  assert(data->getHyper()->sameSize(model->getHyper()));
  assert(data->getHyper()->sameSize(velPadded->getHyper()));

  setDomainRange(model, data);
  _velPadded = velPadded;

  // Initialize J and Laplacian operators
  _Laplacian.reset(new Laplacian2d(model, data));
  _dt = dt;
  _laplTemp.reset(new float2DReg(data->getHyper()));
}

void C2::forward(const bool                         add,
                 const std::shared_ptr<SEP::Vector>model,
                 std::shared_ptr<SEP::Vector>      data)
{
  assert(checkDomainRange(model, data, true));
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;

  std::shared_ptr<float2D> vel =
    ((std::dynamic_pointer_cast<float2DReg>(_velPadded))->_mat);
  std::shared_ptr<float2D> lapl =
    ((std::dynamic_pointer_cast<float2DReg>(_laplTemp))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);

  if (!add) data->scale(0.);

  // data = Lapl(model)
  _Laplacian->forward(0, model, _laplTemp);


  // data = v*v*dt*dt*Lapl(model)+2*model
  #pragma omp parallel for collapse(2)


  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*d)[i2][i1] += _dt * _dt * double((*vel)[i2][i1]) *
                      double((*vel)[i2][i1]) *
                      double((*lapl)[i2][i1]) + 2 * double((*m)[i2][i1]);
    }
  }
}

void C2::adjoint(const bool                         add,
                 std::shared_ptr<SEP::Vector>      model,
                 const std::shared_ptr<SEP::Vector>data)
{
  assert(checkDomainRange(model, data, true));
  int n1 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float2DReg>(model))->getHyper()->getAxis(2).n;

  std::shared_ptr<float2D> vel =
    ((std::dynamic_pointer_cast<float2DReg>(_velPadded))->_mat);
  std::shared_ptr<float2D> lapl =
    ((std::dynamic_pointer_cast<float2DReg>(_laplTemp))->_mat);
  std::shared_ptr<float2D> d =
    ((std::dynamic_pointer_cast<float2DReg>(data))->_mat);
  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);

  if (!add) model->scale(0.);


  // data = v*v*dt*dt*Lapl(model)+2*model
    #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*lapl)[i2][i1] = _dt * _dt * double((*vel)[i2][i1]) *
                        double((*vel)[i2][i1]) *
                        double((*d)[i2][i1]); // + 2 * (*d)[i2][i1];
    }
  }

  // data = Lapl(model)
  _Laplacian->adjoint(add, model, _laplTemp);

  // data = v*v*dt*dt*Lapl(model)+2*model
    #pragma omp parallel for collapse(2)

  for (int i2 = 0; i2 < n2; i2++) {
    for (int i1 = 0; i1 < n1; i1++) {
      (*m)[i2][i1] +=  2 * double((*d)[i2][i1]);
    }
  }
}

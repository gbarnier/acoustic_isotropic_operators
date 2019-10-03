#include <C2_2DCube.h>
using namespace giee;
using namespace waveform;

C2_2DCube::C2_2DCube(
  const std::shared_ptr<SEP::float3DReg>model,
  const std::shared_ptr<SEP::float3DReg>data,
  const std::shared_ptr<SEP::float2DReg>velPadded,
  const float                            dt
  ) {
  // model and data and velocity domains must match
  assert(data->getHyper()->sameSize(model->getHyper()));
  assert(model->getHyper()->getAxis(2).n == velPadded->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(3).n == velPadded->getHyper()->getAxis(2).n);

  setDomainRange(model, data);
  _velPadded = velPadded;

  // Initialize J and Laplacian operators
  _Laplacian.reset(new Laplacian2d(model, data));
  _dt = dt;
  _laplTemp.reset(new float3DReg(data->getHyper()));
}

void C2_2DCube::forward(const bool                         add,
                        const std::shared_ptr<SEP::Vector>model,
                        std::shared_ptr<SEP::Vector>      data)
{
  assert(checkDomainRange(model, data));
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

  std::shared_ptr<float2D> vel =
    ((std::dynamic_pointer_cast<float2DReg>(_velPadded))->_mat);
  std::shared_ptr<float3D> lapl =
    ((std::dynamic_pointer_cast<float3DReg>(_laplTemp))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);

  if (!add) data->scale(0.);

  // data = Lapl(model)
  _Laplacian->forward(0, model, _laplTemp);


  // data = v*v*dt*dt*Lapl(model)+2*model
  #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*d)[i3][i2][i1] += _dt * _dt * (*vel)[i3][i2] * (*vel)[i3][i2] *
                            (*lapl)[i3][i2][i1] + 2 * (*m)[i3][i2][i1];
      }
    }
  }
}

void C2_2DCube::adjoint(const bool                         add,
                        std::shared_ptr<SEP::Vector>      model,
                        const std::shared_ptr<SEP::Vector>data)
{
  assert(checkDomainRange(model, data));
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

  std::shared_ptr<float2D> vel =
    ((std::dynamic_pointer_cast<float2DReg>(_velPadded))->_mat);
  std::shared_ptr<float3D> lapl =
    ((std::dynamic_pointer_cast<float3DReg>(_laplTemp))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);

  if (!add) model->scale(0.);
  #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*lapl)[i3][i2][i1] = _dt * _dt * (*vel)[i3][i2] * (*vel)[i3][i2] *
                              (*d)[i3][i2][i1]; // + 2 * (*d)[i2][i1];
      }
    }
  }

  // data = Lapl(model)
  _Laplacian->adjoint(add, model, _laplTemp);


  // data = v*v*dt*dt*Lapl(model)+2*model
    #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*m)[i3][i2][i1] +=  2 * (*d)[i3][i2][i1];
      }
    }
  }
}

#include <ScaleSourceAcousticMonopole.h>
using namespace giee;
using namespace waveform;

ScaleSourceAcousticMonopole::ScaleSourceAcousticMonopole(
  const std::shared_ptr<SEP::float3DReg>model,
  const std::shared_ptr<SEP::float3DReg>data,
  const std::shared_ptr<SEP::float2DReg>velModel)
{
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(
           2).n == velModel->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(
           2).n == velModel->getHyper()->getAxis(2).n);
  setDomainRange(model, data);

  _velModel = velModel;
  _dt       = model->getHyper()->getAxis(1).d;
}

void ScaleSourceAcousticMonopole::forward(const bool                         add,
                                          const std::shared_ptr<SEP::Vector>model,
                                          std::shared_ptr<SEP::Vector>      data)
{
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);

  const std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  const std::shared_ptr<float2D> vel =
    ((std::dynamic_pointer_cast<float2DReg>(_velModel))->_mat);

  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

  double weight = 1;

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*d)[i3][i2][i1] =  -1 * (*vel)[i3][i2][i1] * (*vel)[i3][i2] * _dt * _dt;
      }
    }
  }
}

void ScaleSourceAcousticMonopole::adjoint(const bool                         add,
                                          std::shared_ptr<SEP::Vector>      model,
                                          const std::shared_ptr<SEP::Vector>data)
{
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);

  std::shared_ptr<float3D> m =
    ((std::dynamic_pointer_cast<float3DReg>(model))->_mat);
  const std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  const std::shared_ptr<float2D> vel =
    ((std::dynamic_pointer_cast<float2DReg>(_velModel))->_mat);

  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(model))->getHyper()->getAxis(3).n;

  double weight = 1;

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*m)[i3][i2][i1] =  -1 * (*vel)[i3][i2][i1] * (*vel)[i3][i2] * _dt * _dt;
      }
    }
  }
}

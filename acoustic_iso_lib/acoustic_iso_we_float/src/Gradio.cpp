#include <Gradio.h>
using namespace WRI;
using namespace giee;

Gradio::Gradio(const std::shared_ptr<giee::float2DReg>model,
               const std::shared_ptr<giee::float3DReg>data,
               const std::shared_ptr<giee::float3DReg>pressureData) {
  // ensure pressureData and Data have same dimensions
  assert(data->getHyper()->getAxis(1).n == pressureData->getHyper()->getAxis(
           1).n &&
         data->getHyper()->getAxis(2).n == pressureData->getHyper()->getAxis(
           2).n &&
         data->getHyper()->getAxis(3).n ==
         pressureData->getHyper()->getAxis(3).n);

  assert(data->getHyper()->getAxis(1).d ==
         pressureData->getHyper()->getAxis(1).d);

  // ensure x locations (2nd dim in data and 3rd dim in data) match
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(3).n);

  // ensure z locations (1st dim in data and 2nd dim in data) match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(2).n);

  setDomainRange(model, data);
  _pressureData = pressureData;
  _pressureDatad2.reset(new float3DReg(data->getHyper()));
  _pressureDatad2->scale(0);
  _dt = pressureData->getHyper()->getAxis(1).d;

  std::shared_ptr<float3D> pd_dt2 =
    ((std::dynamic_pointer_cast<float3DReg>(_pressureDatad2))->_mat);
  std::shared_ptr<float3D> pd =
    ((std::dynamic_pointer_cast<float3DReg>(_pressureData))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n; // nt
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n; // nz
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n; // nx

  // TAKE SECOND TIME DERIVATIVE (EDGE CASES)
  // for nx
  // for nz
  // handle boundary at t=0 and t=nt-1
        #pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      (*pd_dt2)[i3][i2][0] =
        (-2 * (*pd)[i3][i2][0] + (*pd)[i3][i2][1]) / (_dt * _dt);
      (*pd_dt2)[i3][i2][n1 - 1] =
        (-2 * (*pd)[i3][i2][n1 - 1] + (*pd)[i3][i2][n1 - 2]) / (_dt * _dt);
    }
  }

  // TAKE SECOND TIME DERIVATIVE (ALL OTHER CASES)
  // for nx
  // for nz
  // for nt
        #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 1; i1 < n1 - 1; i1++) {
        (*pd_dt2)[i3][i2][i1] =
          ((*pd)[i3][i2][i1 - 1] - 2 * (*pd)[i3][i2][i1] +
           (*pd)[i3][i2][i1 + 1]) /
          (_dt * _dt);
      }
    }
  }
}

void Gradio::forward(const bool                         add,
                     const std::shared_ptr<giee::Vector>model,
                     std::shared_ptr<giee::Vector>      data) {
  assert(checkDomainRange(model, data, true));

  if (!add) data->scale(0.);

  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float3D> pd =
    ((std::dynamic_pointer_cast<float3DReg>(_pressureData))->_mat);
  std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  const std::shared_ptr<float3D> pd_dt2 =
    ((std::dynamic_pointer_cast<float3DReg>(_pressureDatad2))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;

  // loop over data output
  // for nx
  // for nz
  // for nt
    #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*d)[i3][i2][i1] += (*pd_dt2)[i3][i2][i1] * (*m)[i3][i2];
      }
    }
  }
}

void Gradio::adjoint(const bool                         add,
                     std::shared_ptr<giee::Vector>      model,
                     const std::shared_ptr<giee::Vector>data) {
  assert(checkDomainRange(model, data, true));

  if (!add) model->scale(0.);

  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float3D> pd =
    ((std::dynamic_pointer_cast<float3DReg>(_pressureData))->_mat);
  const std::shared_ptr<float3D> d =
    ((std::dynamic_pointer_cast<float3DReg>(data))->_mat);
  const std::shared_ptr<float3D> pd_dt2 =
    ((std::dynamic_pointer_cast<float3DReg>(_pressureDatad2))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(data))->getHyper()->getAxis(3).n;

  // loop over model output
  // for nx
  // for nz
  // for nt
    #pragma omp parallel for collapse(2)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*m)[i3][i2] += (*pd_dt2)[i3][i2][i1] * (*d)[i3][i2][i1];
      }
    }
  }
}

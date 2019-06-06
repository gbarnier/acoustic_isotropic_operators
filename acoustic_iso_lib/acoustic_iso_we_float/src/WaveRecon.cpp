#include <WaveRecon.h>


WaveRecon::WaveRecon(const std::shared_ptr<SEP::float3DReg>model,
                     const std::shared_ptr<SEP::float3DReg>data,
                     const std::shared_ptr<SEP::float2DReg>slowModel,
                     int                                    n1min,
                     int                                    n1max,
                     int                                    n2min,
                     int                                    n2max,
                     int                                    n3min,
                     int                                    n3max,
                     int                                    boundaryCond) {
  // ensure model and data dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  // ensure velModel matches spatial dimensions of model and data
  assert(model->getHyper()->getAxis(2).n ==
         slowModel->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(3).n ==
         slowModel->getHyper()->getAxis(2).n);

  // boundary condition must be 1 or 0
  assert(boundaryCond == 1 || boundaryCond == 0);

  // set domain and range
  setDomainRange(model, data);

  _n1min = n1min;
  _n1max = n1max;

  _n2min = n2min;
  _n2max = n2max;

  _n3min = n3min;
  _n3max = n3max;

  // set up time deriv and lapl operator
  _W.reset(new Mask3d(model, data, _n1min, _n1max, _n2min, _n2max, _n3min,
                      _n3max, boundaryCond));
  _D.reset(new SecondDeriv(model, data));
  _L.reset(new Laplacian2d(model, data));

  // set slowness
  _slowModel = slowModel;
}

// WAm=d
// W[d^2/dt^2(model)*s^2 -Lapl(model)]=data
void WaveRecon::forward(const bool                         add,
                        const std::shared_ptr<SEP::float3DReg>model,
                        std::shared_ptr<SEP::float3DReg>      data) {
  assert(checkDomainRange(model, data, true));

  std::shared_ptr<SEP::float3DReg> temp0 =
    std::dynamic_pointer_cast<float3DReg>(data->clone());
  std::shared_ptr<SEP::float3DReg> temp1 =
    std::dynamic_pointer_cast<float3DReg>(data->clone());
  temp0->scale(0.);
  temp1->scale(0.);

  // get pointer to temp0 values and dimensions
  std::shared_ptr<float3D> t0 =
    ((std::dynamic_pointer_cast<float3DReg>(temp0))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(temp0))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(temp0))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(temp0))->getHyper()->getAxis(3).n;

  // get pointer to slowness values
  std::shared_ptr<float2D> s =
    ((std::dynamic_pointer_cast<float2DReg>(_slowModel))->_mat);

  // forward second derivative
  _D->forward(0, model, temp0); // temp0 = d^2/dt^2(model)

  // multiply by slowness squared
  #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*t0)[i3][i2][i1] = (*t0)[i3][i2][i1] * (*s)[i3][i2] * (*s)[i3][i2];

        // temp0 = d^2/dt^2(model)*s^2
      }
    }
  }

  // subtract forward lapl
  _L->forward(0, model, temp1);  // temp1 = Lapl(model)

  temp0->scaleAdd(1, temp1, -1); // temp 0 = temp0 - temp1 = d^2/dt^2(model)*s^2
                                 // -
                                 // Lapl(model)

  // handle boundaries: indices within laplacian stencil width of boundary
  // should be set to zero. You can think of this as allowing psuedo sources
  // and sinks to exist outside of the boundary that allow these indices to
  // always obey the wave equation.
  _W->forward(0, temp0, temp1); // temp1 = W[temp0] = W[d^2/dt^2(model)*s^2
                                // -Lapl(model)]

  if (!add) data->scaleAdd(0, temp1, 1);
  else {
    data->scaleAdd(1, temp1, 1);
  }
}

// A*W*d=m
// (d^2/dt^2)*(W(data))*s^2 -Lapl*(W(data))]=model
void WaveRecon::adjoint(const bool                         add,
                        std::shared_ptr<SEP::float3DReg>      model,
                        const std::shared_ptr<SEP::float3DReg>data) {
  assert(checkDomainRange(model, data, true));

  std::shared_ptr<SEP::float3DReg> temp0 =
    std::dynamic_pointer_cast<float3DReg>(model->clone());
  std::shared_ptr<SEP::float3DReg> temp1 =
    std::dynamic_pointer_cast<float3DReg>(model->clone());
  std::shared_ptr<SEP::float3DReg> temp2 =
    std::dynamic_pointer_cast<float3DReg>(model->clone());

  temp0->scale(0.);
  temp1->scale(0.);
  temp2->scale(0.);


  // get pointer to data values and dimensions
  std::shared_ptr<float3D> t0 =
    ((std::dynamic_pointer_cast<float3DReg>(temp1))->_mat);
  int n1 =
    (std::dynamic_pointer_cast<float3DReg>(temp1))->getHyper()->getAxis(1).n;
  int n2 =
    (std::dynamic_pointer_cast<float3DReg>(temp1))->getHyper()->getAxis(2).n;
  int n3 =
    (std::dynamic_pointer_cast<float3DReg>(temp1))->getHyper()->getAxis(3).n;


  _W->adjoint(0, temp0, data);  // temp0=W*(data)
  // adjoint second derivative
  _D->adjoint(0, temp1, temp0); // temp1 = (d^2/dt^2)*(W(data))

  // get pointer to slowness values
  std::shared_ptr<float2D> s =
    ((std::dynamic_pointer_cast<float2DReg>(_slowModel))->_mat);

  // multiply by slowness squared
  #pragma omp parallel for collapse(3)

  for (int i3 = 0; i3 < n3; i3++) {
    for (int i2 = 0; i2 < n2; i2++) {
      for (int i1 = 0; i1 < n1; i1++) {
        (*t0)[i3][i2][i1] = (*t0)[i3][i2][i1] * (*s)[i3][i2] * (*s)[i3][i2]; // temp1
                                                                             // =
                                                                             // (d^2/dt^2)*(W(data))s^2
      }
    }
  }

  // calc adjoint lapl
  _L->adjoint(0, temp2, temp0);  // temp2 = Lapl*(W(data))

  // subtract adjoint lapl
  temp1->scaleAdd(1, temp2, -1); // temp1 = temp1 - temp2 =
                                 // (d^2/dt^2)*(W(data))s^2 -//
                                 // Lapl*(W(data))


  if (!add) model->scaleAdd(0, temp1, 1);
  else model->scaleAdd(1, temp1, 1);
}

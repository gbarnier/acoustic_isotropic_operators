#include <Gradio_multi_exp.h>
using namespace SEP;

Gradio_multi_exp::Gradio_multi_exp(const std::shared_ptr<SEP::float2DReg>model,
               const std::shared_ptr<SEP::float4DReg>data,
               const std::shared_ptr<SEP::float4DReg>pressureData) {
  // ensure pressureData and Data have same dimensions
  assert(data->getHyper()->getAxis(1).n == pressureData->getHyper()->getAxis(1).n &&
         data->getHyper()->getAxis(2).n == pressureData->getHyper()->getAxis(2).n &&
         data->getHyper()->getAxis(3).n == pressureData->getHyper()->getAxis(3).n &&
         data->getHyper()->getAxis(4).n == pressureData->getHyper()->getAxis(4).n);

  assert(data->getHyper()->getAxis(1).d == pressureData->getHyper()->getAxis(1).d);

  // ensure x locations (2nd dim in model and 2nd dim in data) match
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);

  // ensure z locations (1st dim in model and 1st dim in data) match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);

  setDomainRange(model, data);
  _pressureData = pressureData;
  _pressureDatad2.reset(new float4DReg(data->getHyper()));
  _pressureDatad2->zero();
  _dt2 = pressureData->getHyper()->getAxis(3).d;
  _dt2*=_dt2;
  _nShots = data->getHyper()->getAxis(4).n;

  std::shared_ptr<float4D> pd_dt2 =
    ((std::dynamic_pointer_cast<float4DReg>(_pressureDatad2))->_mat);
  std::shared_ptr<float4D> pd =
    ((std::dynamic_pointer_cast<float4DReg>(_pressureData))->_mat);
  _n1 =
    (std::dynamic_pointer_cast<float4DReg>(data))->getHyper()->getAxis(1).n; // nt
  _n2 =
    (std::dynamic_pointer_cast<float4DReg>(data))->getHyper()->getAxis(2).n; // nz
  _n3 =
    (std::dynamic_pointer_cast<float4DReg>(data))->getHyper()->getAxis(3).n; // nx

  // TAKE SECOND TIME DERIVATIVE
  std::shared_ptr<SEP::float4DReg>buffer;
  buffer.reset(new SEP::float4DReg(data->getHyper()->getAxis(1).n,
                                    data->getHyper()->getAxis(2).n ,
                                    data->getHyper()->getAxis(3).n + 2*_dt2Order,
                                    data->getHyper()->getAxis(4).n));
  buffer->zero();
  std::shared_ptr<float4D> b =
    ((std::dynamic_pointer_cast<float4DReg>(buffer))->_mat);

  C0t = -2.927222222;
  C1t =  1.666666667;
  C2t = -0.23809524;
  C3t =  0.03968254;
  C4t = -0.00496032;
  C5t =  0.00031746;
    // load buffer
  #pragma omp parallel for collapse(4)
  for (int is = 0; is <_nShots; is++) {
    for (int it = 0; it <_n3; it++) {
      for (int ix = 0; ix <_n2; ix++) {
        for (int iz = 0; iz <_n1; iz++) {
          (*b)[is][it + _dt2Order][ix][iz] = (*pd)[is][it][ix][iz];
        }
      }
    }
  }
   //take second deriv
   #pragma omp parallel for collapse(4)
  for (int is = 0; is <_nShots; is++) {
    for (int it = 0; it <_n3; it++) { //time
      for (int ix = 0; ix <_n2; ix++) { //x
        for (int iz = 0; iz <_n1; iz++) { //z
          (*pd_dt2)[is][it][ix][iz] =       //second time deriv
          			 (C0t* (*b)[is][it+_dt2Order][ix ][iz ] + \
                  C1t * ((*b)[is][it+_dt2Order-1][ix ][iz ]+(*b)[is][it+_dt2Order + 1][ix ][iz]) + \
                  C2t * ((*b)[is][it+_dt2Order-2][ix ][iz ]+(*b)[is][it+_dt2Order + 2][ix ][iz]) + \
                  C3t * ((*b)[is][it+_dt2Order-3][ix ][iz ]+(*b)[is][it+_dt2Order + 3][ix ][iz]) + \
                  C4t * ((*b)[is][it+_dt2Order-4][ix ][iz ]+(*b)[is][it+_dt2Order + 4][ix ][iz]) + \
                  C5t * ((*b)[is][it+_dt2Order-5][ix ][iz ]+(*b)[is][it+_dt2Order + 5][ix ][iz]))/_dt2;
        }
      }
    }
  }
}

void Gradio_multi_exp::forward(const bool                         add,
                       const std::shared_ptr<SEP::float2DReg>model,
                       std::shared_ptr<SEP::float4DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) data->scale(0.);

  const std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float4D> pd =
    ((std::dynamic_pointer_cast<float4DReg>(_pressureData))->_mat);
  std::shared_ptr<float4D> d =
    ((std::dynamic_pointer_cast<float4DReg>(data))->_mat);
  const std::shared_ptr<float4D> pd_dt2 =
    ((std::dynamic_pointer_cast<float4DReg>(_pressureDatad2))->_mat);

  // loop over data output
  // for nt
  // for nx
  // for nz
  #pragma omp parallel for collapse(4)
  for (int is = 0; is <_nShots; is++) {
    for (int i3 = 0; i3 <_n3; i3++) {
      for (int i2 = 0; i2 <_n2; i2++) {
        for (int i1 = 0; i1 <_n1; i1++) {
          (*d)[is][i3][i2][i1] += (*pd_dt2)[is][i3][i2][i1] * (*m)[i2][i1];
        }
      }
    }
  }
}

void Gradio_multi_exp::adjoint(const bool                         add,
                       std::shared_ptr<SEP::float2DReg>model,
                       const std::shared_ptr<SEP::float4DReg>      data) const {
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<float2D> m =
    ((std::dynamic_pointer_cast<float2DReg>(model))->_mat);
  const std::shared_ptr<float4D> pd =
    ((std::dynamic_pointer_cast<float4DReg>(_pressureData))->_mat);
  const std::shared_ptr<float4D> d =
    ((std::dynamic_pointer_cast<float4DReg>(data))->_mat);
  const std::shared_ptr<float4D> pd_dt2 =
    ((std::dynamic_pointer_cast<float4DReg>(_pressureDatad2))->_mat);

  // loop over model output
  // for nz
  // for nx
  // for nt

  #pragma omp parallel for collapse(2)
  for (int i2 = 0; i2 <_n2; i2++) {
    for (int i1 = 0; i1 <_n1; i1++) {
      // CANT BE PARALLEL
      for (int is = 0; is <_nShots; is++) {
        for (int i3 = 0; i3 <_n3; i3++) {
          (*m)[i2][i1] += (*pd_dt2)[is][i3][i2][i1] * (*d)[is][i3][i2][i1];
        }
      }
    }
  }
}

void Gradio_multi_exp::set_wfld(std::shared_ptr<SEP::float4DReg> new_pressureData){
  //create buffer
  std::shared_ptr<SEP::float4DReg>buffer;
  buffer.reset(new SEP::float4DReg(new_pressureData->getHyper()->getAxis(1).n,
                                    new_pressureData->getHyper()->getAxis(2).n ,
                                    new_pressureData->getHyper()->getAxis(3).n + 2*_dt2Order,
                                    new_pressureData->getHyper()->getAxis(4).n));
  buffer->scale(0);

  //get pointers
  std::shared_ptr<float4D> pd_dt2 =
    ((std::dynamic_pointer_cast<float4DReg>(_pressureDatad2))->_mat);
  std::shared_ptr<float4D> pd =
    ((std::dynamic_pointer_cast<float4DReg>(new_pressureData))->_mat);
  std::shared_ptr<float4D> b =
    ((std::dynamic_pointer_cast<float4DReg>(buffer))->_mat);

    // load buffer
  #pragma omp parallel for collapse(4)
  for (int is = 0; is <_nShots; is++) {
    for (int it = 0; it <_n3; it++) {
      for (int ix = 0; ix <_n2; ix++) {
        for (int iz = 0; iz <_n1; iz++) {
          (*b)[is][it + _dt2Order][ix][iz] = (*pd)[is][it][ix][iz];
        }
      }
    }
  }
  //take second deriv
  #pragma omp parallel for collapse(4)
  for (int is = 0; is <_nShots; is++) {
   for (int it = 0; it <_n3; it++) { //time
     for (int ix = 0; ix <_n2; ix++) { //x
       for (int iz = 0; iz <_n1; iz++) { //z
         (*pd_dt2)[is][it][ix][iz] =       //second time deriv
                (C0t* (*b)[is][it+_dt2Order][ix ][iz ] + \
                 C1t * ((*b)[is][it+_dt2Order-1][ix ][iz ]+(*b)[is][it+_dt2Order + 1][ix ][iz]) + \
                 C2t * ((*b)[is][it+_dt2Order-2][ix ][iz ]+(*b)[is][it+_dt2Order + 2][ix ][iz]) + \
                 C3t * ((*b)[is][it+_dt2Order-3][ix ][iz ]+(*b)[is][it+_dt2Order + 3][ix ][iz]) + \
                 C4t * ((*b)[is][it+_dt2Order-4][ix ][iz ]+(*b)[is][it+_dt2Order + 4][ix ][iz]) + \
                 C5t * ((*b)[is][it+_dt2Order-5][ix ][iz ]+(*b)[is][it+_dt2Order + 5][ix ][iz]))/_dt2;
        }
      }
    }
  }
}

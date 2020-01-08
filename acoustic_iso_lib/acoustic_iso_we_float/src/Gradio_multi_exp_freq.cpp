#include <Gradio_multi_exp_freq.h>
using namespace SEP;


Gradio_multi_exp_freq::Gradio_multi_exp_freq(const std::shared_ptr<SEP::float2DReg>model,
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
  _dw = pressureData->getHyper()->getAxis(3).d;
  _ow = pressureData->getHyper()->getAxis(3).o;

  std::shared_ptr<float4D> pd_dt2 =
    ((std::dynamic_pointer_cast<float4DReg>(_pressureDatad2))->_mat);
  std::shared_ptr<float4D> pd =
    ((std::dynamic_pointer_cast<float4DReg>(_pressureData))->_mat);
  _n1 = data->getHyper()->getAxis(1).n; // nz
  _n2 = data->getHyper()->getAxis(2).n; // nx
  _n3 = data->getHyper()->getAxis(3).n; // nw
  _n4 = data->getHyper()->getAxis(4).n; //n Experiments

  // TAKE SECOND TIME DERIVATIVE
   #pragma omp parallel for collapse(4)
  for (int is = 0; is <_n4; is++) {
    for (int iw = 0; iw <_n3; iw++) { //time
      for (int ix = 0; ix <_n2; ix++) { //x
        for (int iz = 0; iz <_n1; iz++) { //z
          float w = _ow + _dw * iw;
          (*pd_dt2)[is][iw][ix][iz] =       //second time deriv
          			 w * w * (*pd)[is][iw][ix][iz] ;
        }
      }
    }
  }
}

void Gradio_multi_exp_freq::forward(const bool                         add,
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
  for (int is = 0; is <_n4; is++) {
    for (int i3 = 0; i3 <_n3; i3++) {
      for (int i2 = 0; i2 <_n2; i2++) {
        for (int i1 = 0; i1 <_n1; i1++) {
          (*d)[is][i3][i2][i1] += (*pd_dt2)[is][i3][i2][i1] * (*m)[i2][i1];
        }
      }
    }
  }
}

void Gradio_multi_exp_freq::adjoint(const bool                         add,
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
      for (int is = 0; is <_n4; is++) {
        for (int i3 = 0; i3 <_n3; i3++) {
          (*m)[i2][i1] += (*pd_dt2)[is][i3][i2][i1] * (*d)[is][i3][i2][i1];
        }
      }
    }
  }
}

void Gradio_multi_exp_freq::set_wfld(std::shared_ptr<SEP::float4DReg> new_pressureData){
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

  // TAKE SECOND TIME DERIVATIVE
   #pragma omp parallel for collapse(4)
  for (int is = 0; is <_n4; is++) {
    for (int iw = 0; iw <_n3; iw++) { //time
      for (int ix = 0; ix <_n2; ix++) { //x
        for (int iz = 0; iz <_n1; iz++) { //z
          float w = _ow + _dw * iw;
          (*pd_dt2)[is][iw][ix][iz] =       //second time deriv
          			 w * w * (*pd)[is][iw][ix][iz] ;
        }
      }
    }
  }
}

#include <WaveRecon_freq_precond.h>
#include <math.h>

/*###########################################################################
                          Multi experiment
/*###########################################################################*/
WaveRecon_freq_multi_exp_precond::WaveRecon_freq_multi_exp_precond(const std::shared_ptr<SEP::complex4DReg>model,
            const std::shared_ptr<SEP::complex4DReg>data,
            const std::shared_ptr<SEP::float2DReg>slsqModel) {

  // ensure model and data dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(4).n == data->getHyper()->getAxis(4).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  // ensure velModel matches fmatial dimensions of model and data
  assert(model->getHyper()->getAxis(1).n ==
         slsqModel->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n ==
         slsqModel->getHyper()->getAxis(2).n);


  n1 =model->getHyper()->getAxis(1).n; //z
   n2 =model->getHyper()->getAxis(2).n; //x
   n3 =model->getHyper()->getAxis(3).n; //hz
   n4 =model->getHyper()->getAxis(4).n; //experiment

  // set domain and range
  setDomainRange(model, data);

  // set up  lapl operator
  // get fmatial sampling
  _db = model->getHyper()->getAxis(1).d;
  _da = model->getHyper()->getAxis(2).d;
  _dw = model->getHyper()->getAxis(3).d;
  _ow = model->getHyper()->getAxis(3).o;

  // calculate lapl coefficients
  C0z = -2.927222222 / (_db * _db);
  C0x = -2.927222222 / (_da * _da);

  setDomainRange(model, data);

  _precond.reset(new SEP::float3DReg(data->getHyper()->getAxis(1).n,
                                    data->getHyper()->getAxis(2).n,
                                    data->getHyper()->getAxis(3).n));
  std::shared_ptr<float2D> s = ((std::dynamic_pointer_cast<float2DReg>(slsqModel))->_mat);
  std::shared_ptr<float3D> p = ((std::dynamic_pointer_cast<float3DReg>(_precond))->_mat);
  #pragma omp parallel for collapse(3)
  for (int iw = 0; iw < n3; iw++) { //freq
    for (int ix = FAT; ix < n2-FAT; ix++) { //x
      for (int iz = FAT; iz < n1-FAT; iz++) { //z
        float w = 2 * M_PI * (_ow + _dw * iw);
        double c = (*s)[ix][iz] * -1 * w * w - C0x - C0z;
        (*p)[iw][ix][iz] = 1 / (c*c);
      }
    }
  }

}

void WaveRecon_freq_multi_exp_precond::update_slsq(std::shared_ptr<SEP::float2DReg>slsq){

  std::shared_ptr<float2D> s = ((std::dynamic_pointer_cast<float2DReg>(slsq))->_mat);
  std::shared_ptr<float3D> p = ((std::dynamic_pointer_cast<float3DReg>(_precond))->_mat);
  #pragma omp parallel for collapse(3)
  for (int iw = 0; iw < n3; iw++) { //freq
    for (int ix = FAT; ix < n2-FAT; ix++) { //x
      for (int iz = FAT; iz < n1-FAT; iz++) { //z
        float w = 2 * M_PI * (_ow + _dw * iw);
        double c = (*s)[ix][iz] * -1 * w * w - C0x - C0z;
        (*p)[iw][ix][iz] = 1 / (c*c);
      }
    }
  }
}

void WaveRecon_freq_multi_exp_precond::forward(const bool                         add,
                          const std::shared_ptr<SEP::complex4DReg>model,
                          std::shared_ptr<SEP::complex4DReg>      data) const {

  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);

  const std::shared_ptr<complex4D> m = ((std::dynamic_pointer_cast<complex4DReg>(model))->_mat);
  std::shared_ptr<complex4D> d = ((std::dynamic_pointer_cast<complex4DReg>(data))->_mat);
  std::shared_ptr<float3D> p = ((std::dynamic_pointer_cast<float3DReg>(_precond))->_mat);
  // std::shared_ptr<float2D> g = _gamma->_mat;
  // std::shared_ptr<float2D> gs = _gammaSq->_mat;

  #pragma omp parallel for collapse(4)
  for(int is = 0; is < n4; is++) { // experiment
    for (int iw = 0; iw < n3; iw++) { //freq
      for (int ix = FAT; ix < n2-FAT; ix++) { //x
        for (int iz = FAT; iz < n1-FAT; iz++) { //z
          (*d)[is][iw][ix][iz] +=
    		    (*m)[is][iw][ix][iz]*(*p)[iw][ix][iz];
        }
      }
    }
  }

}

void WaveRecon_freq_multi_exp_precond::adjoint(const bool                         add,
                          std::shared_ptr<SEP::complex4DReg>      model,
                          const std::shared_ptr<SEP::complex4DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<complex4D> m = ((std::dynamic_pointer_cast<complex4DReg>(model))->_mat);
  const std::shared_ptr<complex4D> d = ((std::dynamic_pointer_cast<complex4DReg>(data))->_mat);
  std::shared_ptr<float3D> p = ((std::dynamic_pointer_cast<float3DReg>(_precond))->_mat);

  #pragma omp parallel for collapse(4)
  for(int is = 0; is < n4; is++) {
    for (int iw = 0; iw < n3; iw++) {
      for (int ix = FAT; ix < n2-FAT; ix++) {
        for (int iz = FAT; iz < n1-FAT; iz++) {
          float w = 2 * M_PI * (_ow + _dw * iw);
          (*m)[is][iw][ix][iz] +=
            (*d)[is][iw][ix][iz]*(*p)[iw][ix][iz];
        }
      }
    }
  }
}

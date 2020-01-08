#include <WaveRecon_freq.h>
#include <math.h>

/*###########################################################################
                          Multi experiment
/*###########################################################################*/
WaveRecon_freq_multi_exp::WaveRecon_freq_multi_exp(const std::shared_ptr<SEP::complex4DReg>model,
                         const std::shared_ptr<SEP::complex4DReg>data,
                         const std::shared_ptr<SEP::float2DReg>slsqModel,
                         float          U_0,
                         float 					alpha,
			                   int					  spongeWidth) {

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

  _spongeWidth=spongeWidth;

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
  C1z =  1.666666667 / (_db * _db);
  C2z = -0.238095238 / (_db * _db);
  C3z =  0.039682539 / (_db * _db);
  C4z = -0.004960317 / (_db * _db);
  C5z =  0.000317460 / (_db * _db);

  C0x = -2.927222222 / (_da * _da);
  C1x =  1.666666667 / (_da * _da);
  C2x = -0.238095238 / (_da * _da);
  C3x =  0.039682539 / (_da * _da);
  C4x = -0.004960317 / (_da * _da);
  C5x =  0.000317460 / (_da * _da);

  setDomainRange(model, data);

    _fatMask.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
                                      data->getHyper()->getAxis(2).n));
    _fatMask->set(0.0);
    for (int ix = FAT; ix < n2-FAT; ix++){
      for (int iz = FAT; iz < n1-FAT; iz++) {
	       (*_fatMask->_mat)[ix][iz] = 1;
      }
    }

    // //calculate gamma
    // _gamma.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
    //                                   data->getHyper()->getAxis(2).n));
    // _gammaSq.reset(new SEP::float2DReg(data->getHyper()->getAxis(1).n,
    //                                   data->getHyper()->getAxis(2).n));
    // _gamma->set(0.0);
    // _gammaSq->set(0.0);
    // _U_0 = U_0;
    // _alpha = alpha;
    // for (int ix = FAT; ix < n2-FAT; ix++){
    //   for (int iz = FAT; iz < n1-FAT; iz++) {
	  //     int distToEdge = std::min(std::min(ix-FAT,iz-FAT),std::min(n1-FAT-iz-1,n2-FAT-ix-1));
	  //     if(distToEdge < _spongeWidth){
    //       float gamma = _U_0/(std::cosh(_alpha*distToEdge)*std::cosh(_alpha*distToEdge));
	  //       (*_gamma->_mat)[ix][iz] = 2*gamma/_dt;
	  //       (*_gammaSq->_mat)[ix][iz] = gamma*gamma;
    //     }
    //   }
    // }
  // set slowness
  _slsq=slsqModel;
}

void WaveRecon_freq_multi_exp::set_slsq(std::shared_ptr<SEP::float2DReg>slsq){
	_slsq=slsq;
}

void WaveRecon_freq_multi_exp::forward(const bool                         add,
                          const std::shared_ptr<SEP::complex4DReg>model,
                          std::shared_ptr<SEP::complex4DReg>      data) const {

  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);

  const std::shared_ptr<complex4D> m = ((std::dynamic_pointer_cast<complex4DReg>(model))->_mat);
  std::shared_ptr<complex4D> d = ((std::dynamic_pointer_cast<complex4DReg>(data))->_mat);
  std::shared_ptr<float2D> fm = _fatMask->_mat;
  std::shared_ptr<float2D> s = ((std::dynamic_pointer_cast<float2DReg>(_slsq))->_mat);
  // std::shared_ptr<float2D> g = _gamma->_mat;
  // std::shared_ptr<float2D> gs = _gammaSq->_mat;

  #pragma omp parallel for collapse(4)
  for(int is = 0; is < n4; is++) { // experiment
    for (int iw = 0; iw < n3; iw++) { //freq
      for (int ix = FAT; ix < n2-FAT; ix++) { //x
        for (int iz = FAT; iz < n1-FAT; iz++) { //z
          float w = 2 * M_PI * (_ow + _dw * iw);
          (*d)[is][iw][ix][iz] +=//second time deriv
    		    (*s)[ix][iz] * -1 * w * w * (*m)[is][iw][ix][iz] -
              //laplacian
              (C0x *(*m)[is][iw][ix ][iz ] + \
              C1x * ((*m)[is][iw][ix + 1 ][iz ] + (*m)[is][iw][ix - 1 ][iz ]) + \
              C2x * ((*m)[is][iw][ix + 2 ][iz ] + (*m)[is][iw][ix - 2 ][iz ]) + \
              C3x * ((*m)[is][iw][ix + 3 ][iz ] + (*m)[is][iw][ix - 3 ][iz ]) + \
              C4x * ((*m)[is][iw][ix + 4 ][iz ] + (*m)[is][iw][ix - 4 ][iz ]) + \
              C5x * ((*m)[is][iw][ix + 5 ][iz ] + (*m)[is][iw][ix - 5 ][iz ]) + \
              C0z * ((*m)[is][iw][ix ][iz ]) + \
              C1z * ((*m)[is][iw][ix ][iz + 1 ] + (*m)[is][iw][ix ][iz - 1 ]) + \
              C2z * ((*m)[is][iw][ix ][iz + 2 ] + (*m)[is][iw][ix ][iz - 2 ]) + \
              C3z * ((*m)[is][iw][ix ][iz + 3 ] + (*m)[is][iw][ix ][iz - 3 ]) + \
              C4z * ((*m)[is][iw][ix ][iz + 4 ] + (*m)[is][iw][ix ][iz - 4 ]) + \
              C5z * ((*m)[is][iw][ix ][iz + 5 ] + (*m)[is][iw][ix ][iz - 5 ]));//*(*fm)[ix][iz];
        }
      }
    }
  }

}

void WaveRecon_freq_multi_exp::adjoint(const bool                         add,
                          std::shared_ptr<SEP::complex4DReg>      model,
                          const std::shared_ptr<SEP::complex4DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<complex4D> m = ((std::dynamic_pointer_cast<complex4DReg>(model))->_mat);
  const std::shared_ptr<complex4D> d = ((std::dynamic_pointer_cast<complex4DReg>(data))->_mat);
  std::shared_ptr<float2D> s = ((std::dynamic_pointer_cast<float2DReg>(_slsq))->_mat);
  std::shared_ptr<float2D> fm = _fatMask->_mat;
  // std::shared_ptr<float2D> g = _gamma->_mat;
  // std::shared_ptr<float2D> gs = _gammaSq->_mat;

  #pragma omp parallel for collapse(4)
  for(int is = 0; is < n4; is++) {
    for (int iw = 0; iw < n3; iw++) {
      for (int ix = FAT; ix < n2-FAT; ix++) {
        for (int iz = FAT; iz < n1-FAT; iz++) {
          float w = 2 * M_PI * (_ow + _dw * iw);
          (*m)[is][iw][ix][iz] += //second time deriv
      		    (*s)[ix][iz] * -1 * w * w * (*d)[is][iw][ix][iz] -
              //laplacian
              (C0x *(*d)[is][iw][ix ][iz ] + \
              C1x * ((*d)[is][iw][ix + 1 ][iz ] + (*d)[is][iw][ix - 1 ][iz ]) + \
              C2x * ((*d)[is][iw][ix + 2 ][iz ] + (*d)[is][iw][ix - 2 ][iz ]) + \
              C3x * ((*d)[is][iw][ix + 3 ][iz ] + (*d)[is][iw][ix - 3 ][iz ]) + \
              C4x * ((*d)[is][iw][ix + 4 ][iz ] + (*d)[is][iw][ix - 4 ][iz ]) + \
              C5x * ((*d)[is][iw][ix + 5 ][iz ] + (*d)[is][iw][ix - 5 ][iz ]) + \
              C0z * ((*d)[is][iw][ix ][iz ]) + \
              C1z * ((*d)[is][iw][ix ][iz + 1 ] + (*d)[is][iw][ix ][iz - 1 ]) + \
              C2z * ((*d)[is][iw][ix ][iz + 2 ] + (*d)[is][iw][ix ][iz - 2 ]) + \
              C3z * ((*d)[is][iw][ix ][iz + 3 ] + (*d)[is][iw][ix ][iz - 3 ]) + \
              C4z * ((*d)[is][iw][ix ][iz + 4 ] + (*d)[is][iw][ix ][iz - 4 ]) + \
              C5z * ((*d)[is][iw][ix ][iz + 5 ] + (*d)[is][iw][ix ][iz - 5 ]));//*(*fm)[ix][iz];
        }
      }
    }
  }
}

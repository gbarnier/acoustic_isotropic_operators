#include <SecondDeriv_multi_exp_freq.h>

SecondDeriv_multi_exp_freq::SecondDeriv_multi_exp_freq(const std::shared_ptr<SEP::complex4DReg>model,
                         const std::shared_ptr<SEP::complex4DReg>data
                         ) {
  // ensure model and data dimensions match
  assert(model->getHyper()->getAxis(1).n == data->getHyper()->getAxis(1).n);
  assert(model->getHyper()->getAxis(2).n == data->getHyper()->getAxis(2).n);
  assert(model->getHyper()->getAxis(3).n == data->getHyper()->getAxis(3).n);
  assert(model->getHyper()->getAxis(4).n == data->getHyper()->getAxis(4).n);
  assert(model->getHyper()->getAxis(1).d == data->getHyper()->getAxis(1).d);
  assert(model->getHyper()->getAxis(2).d == data->getHyper()->getAxis(2).d);
  assert(model->getHyper()->getAxis(3).d == data->getHyper()->getAxis(3).d);

  n1 =model->getHyper()->getAxis(1).n; //z
   n2 =model->getHyper()->getAxis(2).n; //x
   n3 =model->getHyper()->getAxis(3).n; //w
   n4 = model->getHyper()->getAxis(4).n; //experiment

  // set domain and range
  setDomainRange(model, data);

  _dw = model->getHyper()->getAxis(3).d;
  _ow = model->getHyper()->getAxis(3).o;

  setDomainRange(model, data);

}

void SecondDeriv_multi_exp_freq::forward(const bool                         add,
                          const std::shared_ptr<SEP::complex4DReg>model,
                          std::shared_ptr<SEP::complex4DReg>      data) const {
  assert(checkDomainRange(model, data));
  if (!add) data->scale(0.);


  const std::shared_ptr<complex4D> m =
    ((std::dynamic_pointer_cast<complex4DReg>(model))->_mat);
   std::shared_ptr<complex4D> d =
    ((std::dynamic_pointer_cast<complex4DReg>(data))->_mat);

  #pragma omp parallel for collapse(4)
  for(int is = 0; is < n4; is++) { // experiment
    for (int iw = 0; iw < n3; iw++) { //freq
      for (int ix = 0; ix < n2; ix++) { //x
        for (int iz = 0; iz < n1; iz++) { //z
          float w = 2 * M_PI * (_ow + _dw * iw);
          (*d)[is][iw][ix][iz] +=//second time deriv
    		    -1 * w * w * (*m)[is][iw][ix][iz];
        }
      }
    }
  }
}

void SecondDeriv_multi_exp_freq::adjoint(const bool                         add,
                          std::shared_ptr<SEP::complex4DReg>      model,
                          const std::shared_ptr<SEP::complex4DReg>data) const{
  assert(checkDomainRange(model, data));

  if (!add) model->scale(0.);

  std::shared_ptr<complex4D> m =
    ((std::dynamic_pointer_cast<complex4DReg>(model))->_mat);
  const std::shared_ptr<complex4D> d =
    ((std::dynamic_pointer_cast<complex4DReg>(data))->_mat);


  #pragma omp parallel for collapse(4)
  for(int is = 0; is < n4; is++) { // experiment
    for (int iw = 0; iw < n3; iw++) { //freq
      for (int ix = 0; ix < n2; ix++) { //x
        for (int iz = 0; iz < n1; iz++) { //z
          float w = 2 * M_PI * (_ow + _dw * iw);
          (*m)[is][iw][ix][iz] +=//second time deriv
    		    -1 * w * w * (*d)[is][iw][ix][iz];
        }
      }
    }
  }
}

// bool SecondDeriv_multi_exp_freq::dotTest(const bool verbose = false, const float maxError = .00001) const {
//   std::cerr << "cpp dot test not implemented.\n";
// }

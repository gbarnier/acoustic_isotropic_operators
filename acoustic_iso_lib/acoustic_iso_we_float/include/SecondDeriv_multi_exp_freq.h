/********************************************//**
   Author: Stuart Farris
   Date: 04JAN2018
   Description:  2nd derivative operator. Deriv taken over fast axis
      3D Vector
 ***********************************************/

#pragma once
#include <operator.h>
#include <complex4DReg.h>

using namespace SEP;


class SecondDeriv_multi_exp_freq : public Operator<SEP::complex4DReg, SEP::complex4DReg> {
public:

  SecondDeriv_multi_exp_freq(const std::shared_ptr<SEP::complex4DReg>model,
              const std::shared_ptr<SEP::complex4DReg>data);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::complex4DReg>model,
                       std::shared_ptr<SEP::complex4DReg>      data) const;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::complex4DReg>      model,
                       const std::shared_ptr<SEP::complex4DReg>data) const;
  bool dotTest(const bool verbose = false, const float maxError = .00001) const{
    std::cerr << "cpp dot test not implemented.\n";
  }
private:

  float _dw,_ow;
  int n1,n2,n3,n4;
};

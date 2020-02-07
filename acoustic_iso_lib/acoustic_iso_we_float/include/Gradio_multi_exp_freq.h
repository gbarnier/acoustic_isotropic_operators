/********************************************//**
   Author: Stuart Farris
   Date: Oct 2017
   Description:
 ***********************************************/

#pragma once
#include <operator.h>
#include <float2DReg.h>
#include <complex4DReg.h>
 using namespace SEP;
class Gradio_multi_exp_freq : public Operator<SEP::float2DReg, SEP::complex4DReg>{
public:

  Gradio_multi_exp_freq(const std::shared_ptr<SEP::float2DReg>model,
         const std::shared_ptr<SEP::complex4DReg>data,
         const std::shared_ptr<SEP::complex4DReg>pressureData);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float2DReg>model,
                       std::shared_ptr<SEP::complex4DReg>      data) const;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float2DReg>      model,
                       const std::shared_ptr<SEP::complex4DReg>data) const ;

  void set_wfld(std::shared_ptr<SEP::complex4DReg> new_pressureData);

  bool dotTest(const bool verbose = false, const float maxError = .00001) const{
    std::cerr << "cpp dot test not implemented.\n";
  }
private:

  std::shared_ptr<SEP::complex4DReg>_pressureData;
  std::shared_ptr<SEP::complex4DReg>_pressureDatad2;
  float _ow,_dw;
  int _n1,_n2,_n3,_n4;
};

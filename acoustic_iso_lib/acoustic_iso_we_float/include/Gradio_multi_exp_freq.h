/********************************************//**
   Author: Stuart Farris
   Date: Oct 2017
   Description:
 ***********************************************/

#pragma once
#include <operator.h>
#include <float2DReg.h>
#include <float4DReg.h>
 using namespace SEP;
class Gradio_multi_exp_freq : public Operator<SEP::float2DReg, SEP::float4DReg>{
public:

  Gradio_multi_exp_freq(const std::shared_ptr<SEP::float2DReg>model,
         const std::shared_ptr<SEP::float4DReg>data,
         const std::shared_ptr<SEP::float4DReg>pressureData);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float2DReg>model,
                       std::shared_ptr<SEP::float4DReg>      data) const;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float2DReg>      model,
                       const std::shared_ptr<SEP::float4DReg>data) const ;

  void set_wfld(std::shared_ptr<SEP::float4DReg> new_pressureData);
private:

  std::shared_ptr<SEP::float4DReg>_pressureData;
  std::shared_ptr<SEP::float4DReg>_pressureDatad2;
  float _ow,dw;        
  int _n1,_n2,_n3,_n4;
};

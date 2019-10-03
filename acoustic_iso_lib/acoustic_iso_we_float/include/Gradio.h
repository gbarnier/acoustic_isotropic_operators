/********************************************//**
   Author: Stuart Farris
   Date: Oct 2017
   Description:
 ***********************************************/

#pragma once
#include <operator.h>
#include <float2DReg.h>
#include <float3DReg.h>
 using namespace SEP;
class Gradio : public Operator<SEP::float2DReg, SEP::float3DReg>{
public:

  Gradio(const std::shared_ptr<SEP::float2DReg>model,
         const std::shared_ptr<SEP::float3DReg>data,
         const std::shared_ptr<SEP::float3DReg>pressureData);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float2DReg>model,
                       std::shared_ptr<SEP::float3DReg>      data) const;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float2DReg>      model,
                       const std::shared_ptr<SEP::float3DReg>data) const ;

  void set_wfld(std::shared_ptr<SEP::float3DReg> new_pressureData);
private:

  std::shared_ptr<SEP::float3DReg>_pressureData;
  std::shared_ptr<SEP::float3DReg>_pressureDatad2;
  float _dt2;
  int _dt2Order = 5;                                                            // coeff
  int _truncateSize;
  int _n1,_n2,_n3;
  float C0t, C1t, C2t, C3t, C4t, C5t;
};

/********************************************//**
   Author: Stuart Farris
   Date: Oct 2017
   Description:
 ***********************************************/

#pragma once
#include <operator.h>
#include <float2DReg.h>
#include <float3DReg.h>
namespace WRI {
class Gradio : public Operator {
public:

  Gradio(const std::shared_ptr<SEP::float2DReg>model,
         const std::shared_ptr<SEP::float3DReg>data,
         const std::shared_ptr<SEP::float3DReg>pressureData);

  virtual void forward(const bool                         add,
                       const std::shared_ptr<SEP::Vector>model,
                       std::shared_ptr<SEP::Vector>      data);

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<SEP::Vector>      model,
                       const std::shared_ptr<SEP::Vector>data);

private:

  std::shared_ptr<SEP::float3DReg>_pressureData;
  std::shared_ptr<SEP::float3DReg>_pressureDatad2;
  float _dt;
  int _truncateSize;
};
}

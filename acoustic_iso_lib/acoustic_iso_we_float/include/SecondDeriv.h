/********************************************//**
   Author: Stuart Farris
   Date: 04JAN2018
   Description:  2nd derivative operator. Deriv taken over fast axis of 1D, 2D,
      or 3D Vector
 ***********************************************/

#pragma once
#include <Operator.h>
#include <float2DReg.h>
#include <float3DReg.h>
namespace waveform {
class SecondDeriv : public giee::Operator {
public:

  /**
     2d case
   */
  SecondDeriv(const std::shared_ptr<giee::float2DReg>model,
              const std::shared_ptr<giee::float2DReg>data);

  /**
     3d case
   */
  SecondDeriv(const std::shared_ptr<giee::float3DReg>model,
              const std::shared_ptr<giee::float3DReg>data);

  /**
     lapl(model) -> data
   */
  virtual void forward(const bool                         add,
                       const std::shared_ptr<giee::Vector>model,
                       std::shared_ptr<giee::Vector>      data);

  /**
     lapl(data) -> model
   */
  virtual void adjoint(const bool                         add,
                       std::shared_ptr<giee::Vector>      model,
                       const std::shared_ptr<giee::Vector>data);

private:

  float _df2; // sampling of fast axis squared
  int _dim;   // dimensions of input
};
}

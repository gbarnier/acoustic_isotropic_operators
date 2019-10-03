/********************************************//**
   Author: Stuart Farris
   Date: 04JAN2018
   Description:  2nd derivative operator. Deriv taken over fast axis of 1D, 2D,
      or 3D Vector
 ***********************************************/

#pragma once
#include <operator.h>
#include <float2DReg.h>
#include <float3DReg.h>

using namespace SEP;


class SecondDeriv : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  /**
     2d case
   */
  // SecondDeriv(const std::shared_ptr<SEP::float2DReg>model,
  //             const std::shared_ptr<SEP::float2DReg>data);

  /**
     3d case
   */
  SecondDeriv(const std::shared_ptr<SEP::float3DReg>model,
              const std::shared_ptr<SEP::float3DReg>data);

  /**
     lapl(model) -> data
   */
  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float3DReg>model,
                       std::shared_ptr<SEP::float3DReg>      data) const;

  /**
     lapl(data) -> model
   */
  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float3DReg>      model,
                       const std::shared_ptr<SEP::float3DReg>data) const;

private:

  float _df2; // sampling of fast axis squared
  int _dim;   // dimensions of input
};

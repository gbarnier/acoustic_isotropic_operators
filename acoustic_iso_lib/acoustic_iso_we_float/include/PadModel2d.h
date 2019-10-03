/********************************************//**
   Author: Stuart Farris
   Date: 07FEB2018
   Description:  Pad and truncate 2d model by padSize parameter. Options to pad
      with zeros and with edge values.
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float2DReg.h>
using namespace SEP;
class PadModel2d : public Operator<SEP::float2DReg, SEP::float2DReg> {
public:

  // regular grid
  PadModel2d(const std::shared_ptr<SEP::float2DReg>model,
             const std::shared_ptr<SEP::float2DReg>data,
             const int                              padSize   = 1,
             const int                              padOption = 0);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float2DReg>model,
                       std::shared_ptr<SEP::float2DReg>      data) const ;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float2DReg>      model,
                       const std::shared_ptr<SEP::float2DReg>data) const ;

private:

  int _padSize;
  int _padOption;
};

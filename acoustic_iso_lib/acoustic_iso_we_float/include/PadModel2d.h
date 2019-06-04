/********************************************//**
   Author: Stuart Farris
   Date: 07FEB2018
   Description:  Pad and truncate 2d model by padSize parameter. Options to pad
      with zeros and with edge values.
 ***********************************************/
 #pragma once
 #include <Operator.h>
 #include <float2DReg.h>
namespace waveform {
class PadModel2d : public giee::Operator {
public:

  // regular grid
  PadModel2d(const std::shared_ptr<giee::float2DReg>model,
             const std::shared_ptr<giee::float2DReg>data,
             const int                              padSize   = 1,
             const int                              padOption = 0);

  virtual void forward(const bool                         add,
                       const std::shared_ptr<giee::Vector>model,
                       std::shared_ptr<giee::Vector>      data);

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<giee::Vector>      model,
                       const std::shared_ptr<giee::Vector>data);

private:

  int _padSize;
  int _padOption;
};
}

/********************************************//**
   Author: Stuart Farris
   Date: 08FEB2018
   Description:  Mask values in a slice
 ***********************************************/
 #pragma once
 #include <Operator.h>
 #include <float2DReg.h>
namespace waveform {
class Mask2d : public giee::Operator {
public:

  // regular grid
  Mask2d(
    const std::shared_ptr<giee::float2DReg>model,
    const std::shared_ptr<giee::float2DReg>data,
    int                                    n1min,
    int                                    n1max,
    int                                    n2min,
    int                                    n2max,
    int                                    maskType = 0);

  virtual void forward(const bool                         add,
                       const std::shared_ptr<giee::Vector>model,
                       std::shared_ptr<giee::Vector>      data);

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<giee::Vector>      model,
                       const std::shared_ptr<giee::Vector>data);

private:

  int _n1min, _n1max, _n2min, _n2max, _n3min, _n3max;
  int _maskType;
  std::shared_ptr<float2D>_mask;
};
}

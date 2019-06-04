/********************************************//**
   Author: Stuart Farris
   Date: 01MAR2018
   Description:  Pad and truncate 3d model by padSize parameter. Options to pad
      with zeros and with edge values.
 ***********************************************/
 #pragma once
 #include <Operator.h>
 #include <float3DReg.h>
namespace waveform {
class PadModel3d : public giee::Operator {
public:

  // regular grid
  PadModel3d(const std::shared_ptr<giee::float3DReg>model,
             const std::shared_ptr<giee::float3DReg>data,
             const int                              padSize1,
             const int                              padSize2,
             const int                              padSize3,
             const int                              padOption = 0);
  PadModel3d(const std::shared_ptr<giee::float3DReg>model,
             const std::shared_ptr<giee::float3DReg>data,
             const int                              padSize   = 1,
             const int                              padOption = 0);

  virtual void forward(const bool                         add,
                       const std::shared_ptr<giee::Vector>model,
                       std::shared_ptr<giee::Vector>      data);

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<giee::Vector>      model,
                       const std::shared_ptr<giee::Vector>data);

private:

  int _padSize1, _padSize2, _padSize3;
  int _padOption;
};
}

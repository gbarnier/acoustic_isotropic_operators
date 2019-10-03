/********************************************//**
   Author: Stuart Farris
   Date: 01MAR2018
   Description:  Pad and truncate 3d model by padSize parameter. Options to pad
      with zeros and with edge values.
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float3DReg.h>
using namespace SEP;
class PadModel3d : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  // regular grid
  PadModel3d(const std::shared_ptr<SEP::float3DReg>model,
             const std::shared_ptr<SEP::float3DReg>data,
             const int                              padSize1,
             const int                              padSize2,
             const int                              padSize3,
             const int                              padOption = 0);
  PadModel3d(const std::shared_ptr<SEP::float3DReg>model,
             const std::shared_ptr<SEP::float3DReg>data,
             const int                              padSize   = 1,
             const int                              padOption = 0);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float3DReg>model,
                       std::shared_ptr<SEP::float3DReg>      data) const ;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float3DReg>      model,
                       const std::shared_ptr<SEP::float3DReg>data) const ;

private:

  int _padSize1, _padSize2, _padSize3;
  int _padOption;
};

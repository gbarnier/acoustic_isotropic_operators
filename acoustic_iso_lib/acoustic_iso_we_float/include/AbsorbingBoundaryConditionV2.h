/********************************************//**
   Author: Stuart Farris
   Date: Summer 2017
   Description:  Absorbing boundary condition as written in Almomin's SEP149
      report. Calculates and applies absorbing weight on many 2D slices.
 ***********************************************/
#pragma once
#include <BoundaryCondition.h>
#include <float2DReg.h>
#include <float3DReg.h>
using namespace giee;

namespace waveform {
class AbsorbingBoundaryConditionV2 : public waveform::BoundaryCondition {
public:

  AbsorbingBoundaryConditionV2(const std::shared_ptr<giee::float3DReg>model,
                               const std::shared_ptr<giee::float3DReg>data,
                               const std::shared_ptr<giee::float2DReg>paddedVel,
                               const int                              velPadx,
                               const int                              velPadz,
                               const float                            absConst,
                               const float                            dt);

  void forward(const bool                         add,
               const std::shared_ptr<giee::Vector>model,
               std::shared_ptr<giee::Vector>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<giee::Vector>      model,
               const std::shared_ptr<giee::Vector>data);

  // std::shared_ptr<giee::float2DReg> getWeight();

private:

  float _absConst, _dt;

  // std::shared_ptr<giee::float2DReg> _paddedVel;
};
}

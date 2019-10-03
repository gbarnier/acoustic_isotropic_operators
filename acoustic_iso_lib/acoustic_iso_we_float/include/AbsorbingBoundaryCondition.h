/********************************************//**
   Author: Stuart Farris
   Date: Summer 2017
   Description:  Absorbing boundary condition as written in Almomin's SEP149
      report.
 ***********************************************/
#pragma once
#include <BoundaryCondition.h>
#include <float2DReg.h>
#include <float3DReg.h>
using namespace SEP;

class AbsorbingBoundaryCondition : public BoundaryCondition {
public:

  AbsorbingBoundaryCondition(const std::shared_ptr<SEP::float2DReg>model,
                             const std::shared_ptr<SEP::float2DReg>data,
                             const std::shared_ptr<SEP::float2DReg>paddedVel,
                             const int                              velPadx,
                             const int                              velPadz,
                             const float                            absConst,
                             const float                            dt);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::float2DReg>model,
               std::shared_ptr<SEP::float2DReg>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float2DReg>      model,
               const std::shared_ptr<SEP::float2DReg>data);

  // std::shared_ptr<SEP::float2DReg> getWeight();

private:

  float _absConst, _dt;

  // std::shared_ptr<SEP::float2DReg> _paddedVel;
};

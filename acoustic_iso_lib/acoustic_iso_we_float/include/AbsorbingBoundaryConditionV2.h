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

class AbsorbingBoundaryConditionV2 : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  AbsorbingBoundaryConditionV2(const std::shared_ptr<SEP::float3DReg>model,
                               const std::shared_ptr<SEP::float3DReg>data,
                               const std::shared_ptr<SEP::float2DReg>paddedVel,
                               const int                              velPadx,
                               const int                              velPadz,
                               const float                            absConst,
                               const float                            dt);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::float3DReg>model,
               std::shared_ptr<SEP::float3DReg>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float3DReg>      model,
               const std::shared_ptr<SEP::float3DReg>data);

   std::shared_ptr<SEP::float2DReg>getWeight() {
     return _w;
   }

 protected:

   std::shared_ptr<SEP::hypercube>_paddedModelHyper;
   int _velPadx, _velPadz;
   std::shared_ptr<SEP::float2DReg>_w;

private:

  float _absConst, _dt;

  // std::shared_ptr<SEP::float2DReg> _paddedVel;
};

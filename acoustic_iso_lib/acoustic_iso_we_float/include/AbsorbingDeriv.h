#pragma once
#include <operator.h>
#include <float2DReg.h>
#include <float3DReg.h>

using namespace SEP;

class AbsorbingDeriv : public Operator<SEP::float2DReg, SEP::float2DReg> {
public:

  AbsorbingDeriv(const std::shared_ptr<SEP::float2DReg>model,
                 const std::shared_ptr<SEP::float2DReg>data,
                 const int                              velPadx,
                 const int                              velPadz);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::float2DReg>model,
               std::shared_ptr<SEP::float2DReg>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float2DReg>      model,
               const std::shared_ptr<SEP::float2DReg>data);

private:

  int _velPadx;
  int _velPadz;
};
class AbsorbingDeriv_2DCube : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  AbsorbingDeriv_2DCube(const std::shared_ptr<SEP::float3DReg>model,
                        const std::shared_ptr<SEP::float3DReg>data,
                        const int                              velPadx,
                        const int                              velPadz);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::float3DReg>model,
               std::shared_ptr<SEP::float3DReg>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float3DReg>      model,
               const std::shared_ptr<SEP::float3DReg>data);

private:

  int _velPadx;
  int _velPadz;
};

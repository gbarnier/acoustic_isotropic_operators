#pragma once
#include <operator.h>
#include <float2DReg.h>
#include <float3DReg.h>
namespace waveform {
class AbsorbingDeriv_2DCube : public Operator {
public:

  AbsorbingDeriv_2DCube(const std::shared_ptr<SEP::float3DReg>model,
                        const std::shared_ptr<SEP::float3DReg>data,
                        const int                              velPadx,
                        const int                              velPadz);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::Vector>model,
               std::shared_ptr<SEP::Vector>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::Vector>      model,
               const std::shared_ptr<SEP::Vector>data);

private:

  int _velPadx;
  int _velPadz;
};
}

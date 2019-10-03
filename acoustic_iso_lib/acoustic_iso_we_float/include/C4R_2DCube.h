#pragma once
#include <operator.h>
#include <float2DReg.h>
#include <float3DReg.h>
using namespace giee;

namespace waveform {
class C4R_2DCube : public Operator {
public:

  C4R_2DCube(
    const std::shared_ptr<SEP::float3DReg>model,
    const std::shared_ptr<SEP::float3DReg>data,
    const std::shared_ptr<SEP::float2DReg>velPadded,
    const int                              velPadx,
    const int                              velPadz,
    const float                            dt
    );

  void forward(const bool                         add,
               const std::shared_ptr<SEP::Vector>model,
               std::shared_ptr<SEP::Vector>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::Vector>      model,
               const std::shared_ptr<SEP::Vector>data);

private:

  void setWeight();

  std::shared_ptr<SEP::float2DReg>_velPadded;
  int _velPadx, _velPadz;
  float _dt;
  std::shared_ptr<SEP::float2DReg>_aborbWeight;
  float const _absConst = 0.15;
};
}

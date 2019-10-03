#pragma once
#include <operator.h>
#include <C2_2DCube.h>
#include <G_2DCube.h>
#include <float2DReg.h>
#include <float3DReg.h>
using namespace giee;
using namespace waveform;

namespace waveform {
class C5_2DCube : public Operator {
public:

  C5_2DCube(
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

  std::shared_ptr<waveform::C2_2DCube>_C2;
  std::shared_ptr<waveform::G_2DCube>_G;
  std::shared_ptr<SEP::float2DReg>_velPadded;
  int _velPadx, _velPadz;
  float _dt;
  std::shared_ptr<SEP::float2DReg>_absorbWeight;
  std::shared_ptr<SEP::float3DReg>_temp0;
  std::shared_ptr<SEP::float3DReg>_temp1;
  float const _absConst = 0.15;
};
}

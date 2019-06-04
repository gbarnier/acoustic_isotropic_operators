#pragma once
#include <Operator.h>
#include <float2DReg.h>
#include <float3DReg.h>
#include <C4R_2DCube.h>
#include <C5_2DCube.h>
#include <C6_2DCube.h>

using namespace giee;

namespace waveform {
class HelmABC : public giee::Operator {
public:

  HelmABC(
    const std::shared_ptr<giee::float3DReg>model,
    const std::shared_ptr<giee::float3DReg>data,
    const std::shared_ptr<giee::float2DReg>velPadded,
    const int                              velPadx,
    const int                              velPadz
    );

  virtual void forward(const bool                         add,
                       const std::shared_ptr<giee::Vector>model,
                       std::shared_ptr<giee::Vector>      data);

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<giee::Vector>      model,
                       const std::shared_ptr<giee::Vector>data);

private:

  std::shared_ptr<waveform::C4R_2DCube>_C4R;
  std::shared_ptr<waveform::C5_2DCube>_C5;
  std::shared_ptr<waveform::C6_2DCube>_C6;
  std::shared_ptr<giee::float3DReg>_C4f;
  std::shared_ptr<giee::float3DReg>_temp0;
  std::shared_ptr<giee::float3DReg>_temp1;
  std::shared_ptr<giee::float3DReg>_temp2;
  float _dt;
};
}
                                          

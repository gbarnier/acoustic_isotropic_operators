#pragma once
#include <Operator.h>
#include <float2DReg.h>
#include <float3DReg.h>

#include <C4_2DCube.h>
#include <C4.h>
#include <C5.h>
#include <C6.h>

using namespace giee;

namespace waveform {
class AcousticProp : public giee::Operator {
public:

  AcousticProp(
    const std::shared_ptr<giee::float3DReg>model,
    const std::shared_ptr<giee::float3DReg>data,
    const std::shared_ptr<giee::float2DReg>velPadded
    );

  virtual void forward(const bool                         add,
                       const std::shared_ptr<giee::Vector>model,
                       std::shared_ptr<giee::Vector>      data);

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<giee::Vector>      model,
                       const std::shared_ptr<giee::Vector>data);

private:

  // std::shared_ptr<waveform::C4_2DCube>_C4;
  // std::shared_ptr<waveform::C5>_C5;
  std::shared_ptr<waveform::C2>_C2;

  // std::shared_ptr<giee::float3DReg>_C4f;
  std::shared_ptr<giee::float2DReg>_temp0;
  std::shared_ptr<giee::float2DReg>_temp1;
  std::shared_ptr<giee::float2DReg>_temp2;
  float _dt;
  int _nt;
};
}

// PURELY VIRTUAL

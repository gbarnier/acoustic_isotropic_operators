#pragma once
#include <Operator.h>
#include <Laplacian2d.h>
#include <float2DReg.h>
#include <float3DReg.h>
using namespace giee;
using namespace waveform;

namespace waveform {
class C2_2DCube : public giee::Operator {
public:

  C2_2DCube(
    const std::shared_ptr<giee::float3DReg>model,
    const std::shared_ptr<giee::float3DReg>data,
    const std::shared_ptr<giee::float2DReg>velPadded,
    const float                            dt
    );

  void forward(const bool                         add,
               const std::shared_ptr<giee::Vector>model,
               std::shared_ptr<giee::Vector>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<giee::Vector>      model,
               const std::shared_ptr<giee::Vector>data);

private:

  std::shared_ptr<waveform::Laplacian2d>_Laplacian;
  std::shared_ptr<giee::float2DReg>_velPadded;
  float _dt;
  std::shared_ptr<giee::float3DReg>_laplTemp;
};
}

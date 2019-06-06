#pragma once
#include <operator.h>
#include <float1DReg.h>
#include <float2DReg.h>
#include <float3DReg.h>
#include <BoudaryCondition.h>
#include <PropagateStepper.h>

using namespace giee;

namespace waveform {
class PropogateAcoustic : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  PropogateAcoustic(const std::shared_ptr<SEP::float3DReg>           model,
                    const std::shared_ptr<SEP::float3DReg>           data,
                    const std::shared_ptr<waveform::PropagateStepper> StepperOp,
                    const std::shared_ptr<waveform::BoundaryCondition>BoundaryOp);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::Vector>model,
               std::shared_ptr<SEP::Vector>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::Vector>      model,
               const std::shared_ptr<SEP::Vector>data);

private:

  std::shared_ptr<float2DReg>_velPadded;
  int _velPadx;
  int _velPadz;
  std::shared_ptr<waveform::PropagateStepper>_Stepper,
  std::shared_ptr<waveform::BoundaryCondition>_BoundaryOp
  waveform::BoundaryCondition _BoundaryOp;
  waveform::PropogateStepper _StepperOp;
  waveform::ScaleSource _ScaleSourceOp;
};
}

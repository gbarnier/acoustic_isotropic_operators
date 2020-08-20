#pragma once
#include <operator.h>
#include <float1DReg.h>
#include <float3DReg.h>
namespace waveform {
class ScaleSourceMonopoleAcoustic : public ScaleSource {
public:

  ScaleSourceMonopoleAcoustic(const std::shared_ptr<SEP::float3DReg>model,
                              const std::shared_ptr<SEP::float3DReg>data,
                              const std::shared_ptr<SEP::float2DReg>velModel);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::Vector>model,
               std::shared_ptr<SEP::Vector>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::Vector>      model,
               const std::shared_ptr<SEP::Vector>data);

private:

  std::shared_ptr<SEP::float2DReg>_velModel;
  float _dt;
};
}

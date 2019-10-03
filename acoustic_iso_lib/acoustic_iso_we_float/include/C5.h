#pragma once
#include <operator.h>
#include <C2.h>
#include <G.h>
#include <float2DReg.h>
using namespace giee;
using namespace waveform;

namespace waveform {
class C5 : public Operator {
public:

  C5(
    const std::shared_ptr<SEP::float2DReg>model,
    const std::shared_ptr<SEP::float2DReg>data,
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

  std::shared_ptr<waveform::C2>_C2;
  std::shared_ptr<waveform::G>_G;
  std::shared_ptr<SEP::float2DReg>_velPadded;
  int _velPadx, _velPadz;
  double _dt;
  std::shared_ptr<SEP::float2DReg>_aborbWeight;
  std::shared_ptr<SEP::float2DReg>_temp0;
  std::shared_ptr<SEP::float2DReg>_temp1;
  double const _absConst = 0.15;
};
}

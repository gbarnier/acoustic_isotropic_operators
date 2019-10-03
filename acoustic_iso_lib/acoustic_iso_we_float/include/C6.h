#pragma once
#include <operator.h>
#include <float2DReg.h>
using namespace giee;

namespace waveform {
class C6 : public Operator {
public:

  C6(
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

  std::shared_ptr<SEP::float2DReg>_velPadded;
  int _velPadx, _velPadz;
  double _dt;
  std::shared_ptr<SEP::float2DReg>_aborbWeight;
  double const _absConst = 0.15;
};
}

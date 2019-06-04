#pragma once
#include <Operator.h>
#include <float2DReg.h>
namespace waveform {
class G : public giee::Operator {
public:

  G(const std::shared_ptr<giee::float2DReg>model,
    const std::shared_ptr<giee::float2DReg>data,
    const int                              velPadx,
    const int                              velPadz);

  void forward(const bool                         add,
               const std::shared_ptr<giee::Vector>model,
               std::shared_ptr<giee::Vector>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<giee::Vector>      model,
               const std::shared_ptr<giee::Vector>data);

private:

  int _velPadx;
  int _velPadz;
};
}

#pragma once
#include <Operator.h>
#include <float2DReg.h>
using namespace giee;

namespace waveform {
class BoundaryCondition : public giee::Operator {
public:

  // BoundaryCondition() {}

  virtual void forward(const bool                         add,
                       const std::shared_ptr<giee::Vector>model,
                       std::shared_ptr<giee::Vector>      data) = 0;

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<giee::Vector>      model,
                       const std::shared_ptr<giee::Vector>data) = 0;

  std::shared_ptr<giee::float2DReg>getWeight() {
    return _w;
  }

protected:

  std::shared_ptr<SEP::hypercube>_paddedModelHyper;
  int _velPadx, _velPadz;
  std::shared_ptr<giee::float2DReg>_w;
};
}

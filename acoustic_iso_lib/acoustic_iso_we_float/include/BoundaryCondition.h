#pragma once
#include <operator.h>
#include <float2DReg.h>
using namespace SEP;

class BoundaryCondition : public Operator<SEP::float2DReg, SEP::float2DReg> {
public:

  // BoundaryCondition() {}

  virtual void forward(const bool                         add,
                       const std::shared_ptr<SEP::float2DReg>model,
                       std::shared_ptr<SEP::float2DReg>      data) = 0;

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float2DReg>      model,
                       const std::shared_ptr<SEP::float2DReg>data) = 0;

  std::shared_ptr<SEP::float2DReg>getWeight() {
    return _w;
  }

protected:

  std::shared_ptr<SEP::hypercube>_paddedModelHyper;
  int _velPadx, _velPadz;
  std::shared_ptr<SEP::float2DReg>_w;
};

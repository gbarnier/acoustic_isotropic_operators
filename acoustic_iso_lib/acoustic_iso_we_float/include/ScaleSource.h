#pragma once
#include <Operator.h>
#include <float1DReg.h>
#include <float3DReg.h>
namespace waveform{
class ScaleSource: public giee::Operator{
  public:

  virtual void forward(const bool add, const std::shared_ptr<giee::Vector> model,
    std::shared_ptr<giee::Vector> data);

  virtual void adjoint(const bool add,  std::shared_ptr<giee::Vector> model,
    const std::shared_ptr<giee::Vector> data);

};
}

//PURELY VIRTUAL

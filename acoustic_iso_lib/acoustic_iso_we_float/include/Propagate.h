#pragma once
#include <operator.h>
#include <float1DReg.h>
#include <float3DReg.h>
namespace waveform {
class Propogate : public Operator {
public:

virtual void forward(const bool add, const std::shared_ptr<SEP::Vector> model,
                     std::shared_ptr<SEP::Vector> data)=0;

virtual void adjoint(const bool add,  std::shared_ptr<SEP::Vector> model,
                     const std::shared_ptr<SEP::Vector> data)=0;

private:

};
}

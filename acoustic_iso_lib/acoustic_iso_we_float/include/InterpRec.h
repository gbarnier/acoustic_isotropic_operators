#pragma once
#include <Operator.h>
#include <float1DReg.h>
#include <float2DReg.h>
namespace waveform {
class InterpRec : public giee::Operator {
public:

  InterpRec(const std::shared_ptr<giee::float2DReg>model,
            const std::shared_ptr<giee::float2DReg>data,
            const std::shared_ptr<giee::float1DReg>dataCoordinates,
            float                                  oversamp);

  void forward(const bool                         add,
               const std::shared_ptr<giee::Vector>model,
               std::shared_ptr<giee::Vector>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<giee::Vector>      model,
               const std::shared_ptr<giee::Vector>data);

private:

  float _o1;
  float _d1;
  float _scale;
  std::shared_ptr<giee::float1DReg>_dataCoordinates;
};
}

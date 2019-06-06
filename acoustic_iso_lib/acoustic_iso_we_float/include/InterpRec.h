#pragma once
#include <operator.h>
#include <float1DReg.h>
#include <float2DReg.h>

class InterpRec : public Operator<SEP::float2DReg, SEP::float2DReg> {
public:

  InterpRec(const std::shared_ptr<SEP::float2DReg>model,
            const std::shared_ptr<SEP::float2DReg>data,
            const std::shared_ptr<SEP::float1DReg>dataCoordinates,
            float                                  oversamp);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::float2DReg>model,
               std::shared_ptr<SEP::float2DReg>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float2DReg>      model,
               const std::shared_ptr<SEP::float2DReg>data);

private:

  float _o1;
  float _d1;
  float _scale;
  std::shared_ptr<SEP::float1DReg>_dataCoordinates;
};

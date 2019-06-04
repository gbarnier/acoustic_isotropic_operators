/********************************************//**
   Author: Stuart Farris
   Date: 06NOV2017
   Description:  2D Laplcian to the tenth order. If 3D cube is passed in, 2D
      lapl is taken for each slice of fast axis.
 ***********************************************/

#pragma once
#include <Operator.h>
#include <float2DReg.h>
#include <float3DReg.h>
namespace waveform {
class Laplacian2dFast : public giee::Operator {
public:

  /**
     2d lapl of each slice of fast axis
   */
  Laplacian2dFast(const std::shared_ptr<giee::float3DReg>model,
                  const std::shared_ptr<giee::float3DReg>data);

  Laplacian2dFast(const std::shared_ptr<giee::float2DReg>model,
                  const std::shared_ptr<giee::float2DReg>data);

  /**
     lapl(model) -> data
   */
  virtual void forward(const bool                         add,
                       const std::shared_ptr<giee::Vector>model,
                       std::shared_ptr<giee::Vector>      data);

  /**
     lapl(data) -> model
   */
  virtual void adjoint(const bool                         add,
                       std::shared_ptr<giee::Vector>      model,
                       const std::shared_ptr<giee::Vector>data);

private:

  float _da, _db;                                                   // spatial
                                                                    // sampling
                                                                    // of two
                                                                    // axis
  bool _3d;                                                         // 3d flag
  float C0z, C1z, C2z, C3z, C4z, C5z, C0x, C1x, C2x, C3x, C4x, C5x; // lapl
                                                                    // coeff
  int _bufferSize = 5;
};
}

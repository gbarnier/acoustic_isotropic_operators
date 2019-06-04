/********************************************//**
   Author: Stuart Farris
   Date: 04JAN2018
   Description:  Wavefield reconstruction operator that solves wave equation for
      an optimal wavefield given a velocity model when the source term is zero.
 ***********************************************/
 #pragma once
 #include <Operator.h>
 #include <float2DReg.h>
 #include <float3DReg.h>
 #include <SecondDeriv.h>
 #include <Laplacian2d.h>
 #include <Mask3d.h>
namespace waveform {
class WaveRecon : public giee::Operator {
public:

  WaveRecon(const std::shared_ptr<giee::float3DReg>model,
            const std::shared_ptr<giee::float3DReg>data,
            const std::shared_ptr<giee::float2DReg>slowModel,
            int                                    n1min,
            int                                    n1max,
            int                                    n2min,
            int                                    n2max,
            int                                    n3min,
            int                                    n3max,
            int                                    boundaryCond = 0);

  WaveRecon(const std::shared_ptr<giee::float3DReg>model,
            const std::shared_ptr<giee::float3DReg>data,
            const std::shared_ptr<giee::float2DReg>slowModel) :
    WaveRecon(model,
              data,
              slowModel,
              0,
              model->getHyper()->getAxis(1).n,
              0,
              model->getHyper()->getAxis(2).n,
              0,
              model->getHyper()->getAxis(3).n,
              0) {}

  void forward(const bool                         add,
               const std::shared_ptr<giee::Vector>model,
               std::shared_ptr<giee::Vector>      data);

  void adjoint(const bool                         add,
               std::shared_ptr<giee::Vector>      model,
               const std::shared_ptr<giee::Vector>data);

private:

  std::shared_ptr<giee::float2DReg>_slowModel;
  std::shared_ptr<waveform::SecondDeriv>_D;
  std::shared_ptr<waveform::Laplacian2d>_L;
  std::shared_ptr<waveform::Mask3d>_W;
  int _laplSize = 5;
  int _n1min, _n1max, _n2min, _n2max, _n3min, _n3max;
};
}

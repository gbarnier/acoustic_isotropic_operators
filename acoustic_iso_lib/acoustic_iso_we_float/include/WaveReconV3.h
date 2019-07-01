/********************************************//**
   Author: Stuart Farris
   Date: 23MAR2018
   Description:  Second iteration of wavefield reconstruction operator that
      solves wave equation for
      an optimal wavefield given a velocity model when the source term is zero.
         Parameterized with slowness SQUARED.
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float2DReg.h>
 #include <float3DReg.h>
 #include <SecondDeriv.h>
 #include <Laplacian2d.h>
 #include <Mask3d.h>

 using namespace SEP;


class WaveReconV3 : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  WaveReconV3(const std::shared_ptr<SEP::float3DReg>model,
              const std::shared_ptr<SEP::float3DReg>data,
              const std::shared_ptr<SEP::float2DReg>slsqModel,
              int                                    n1min,
              int                                    n1max,
              int                                    n2min,
              int                                    n2max,
              int                                    n3min,
              int                                    n3max,
              int                                    boundaryCond = 0);

  WaveReconV3(const std::shared_ptr<SEP::float3DReg>model,
              const std::shared_ptr<SEP::float3DReg>data,
              const std::shared_ptr<SEP::float2DReg>slsqModel) :
    WaveReconV3(model,
                data,
                slsqModel,
                0,
                model->getHyper()->getAxis(1).n,
                0,
                model->getHyper()->getAxis(2).n,
                0,
                model->getHyper()->getAxis(3).n,
                0) {}

  void forward(const bool                         add,
               const std::shared_ptr<SEP::float3DReg>model,
               std::shared_ptr<SEP::float3DReg>      data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float3DReg>      model,
               const std::shared_ptr<SEP::float3DReg>data) const;

private:

  int n1,n2,n3;
  std::shared_ptr<SEP::float2DReg>_slsq_dt2;
  float _da, _db;                                                   // spatial
                                                                    // sampling
                                                                    // of two
                                                                    // axis
  bool _3d;                                                         // 3d flag
  float C0z, C1z, C2z, C3z, C4z, C5z, C0x, C1x, C2x, C3x, C4x, C5x; // lapl
  float C0t, C1t, C2t, C3t, C4t, C5t;

  int _dt2Order = 5;                                                            // coeff
  int _laplOrder = 5;
  std::shared_ptr<SEP::float3DReg>buffer;

  int _laplSize = 5;
  int _n1min, _n1max, _n2min, _n2max, _n3min, _n3max;
  float _dt2;
};

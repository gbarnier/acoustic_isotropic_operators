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

 using namespace SEP;


class WaveReconV5 : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  WaveReconV5(const std::shared_ptr<SEP::float3DReg>model,
              const std::shared_ptr<SEP::float3DReg>data,
              const std::shared_ptr<SEP::float2DReg>slsqModel,
              int                                    boundaryCond,
	      int				     spongeWidth);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::float3DReg>model,
               std::shared_ptr<SEP::float3DReg>      data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float3DReg>      model,
               const std::shared_ptr<SEP::float3DReg>data) const;
private:

  int n1,n2,n3;
  std::shared_ptr<SEP::float2DReg>_slsq;
  std::shared_ptr<SEP::float2DReg>_sponge;
  int _spongeWidth;
  float _da, _db;                                                
  bool _3d;                                                         
  float C0z, C1z, C2z, C3z, C4z, C5z, C0x, C1x, C2x, C3x, C4x, C5x;
  float C0t, C1t, C2t, C3t, C4t, C5t;

  int FAT = 5;

  float _dt;
};

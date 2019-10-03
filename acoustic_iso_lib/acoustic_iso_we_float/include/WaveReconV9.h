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


class WaveReconV9 : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  WaveReconV9(const std::shared_ptr<SEP::float3DReg>model,
              const std::shared_ptr<SEP::float3DReg>data,
              const std::shared_ptr<SEP::float2DReg>slsqModel,
              float                                     U_0,
              float 					alpha,
	      int				     spongeWidth);

  void forward(const bool                         add,
               const std::shared_ptr<SEP::float3DReg>model,
               std::shared_ptr<SEP::float3DReg>      data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float3DReg>      model,
               const std::shared_ptr<SEP::float3DReg>data) const;

  void set_slsq(std::shared_ptr<SEP::float2DReg>slsq);
private:

  int n1,n2,n3;
  std::shared_ptr<SEP::float2DReg>_slsq;
  std::shared_ptr<SEP::float2DReg>_gamma;
  std::shared_ptr<SEP::float2DReg>_gammaSq;
  std::shared_ptr<SEP::float2DReg>_fatMask;
  int _spongeWidth;
  float _U_0, _alpha;
  float _da, _db;                                                
  bool _3d;                                                         
  float C0z, C1z, C2z, C3z, C4z, C5z, C0x, C1x, C2x, C3x, C4x, C5x;
  float C0t_10, C1t_10, C2t_10, C3t_10, C4t_10, C5t_10;
  float C0t_8, C1t_8, C2t_8, C3t_8, C4t_8;
  float C0t_6, C1t_6, C2t_6, C3t_6;
  float C0t_4, C1t_4, C2t_4;
  float C0t_2, C1t_2;

  int FAT = 5;

  float _dt;
};


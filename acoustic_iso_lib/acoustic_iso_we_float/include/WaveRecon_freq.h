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
 #include <complex4DReg.h>

 using namespace SEP;



class WaveRecon_freq_multi_exp : public Operator<SEP::complex4DReg, SEP::complex4DReg> {
public:

  WaveRecon_freq_multi_exp(const std::shared_ptr<SEP::complex4DReg>model,
              const std::shared_ptr<SEP::complex4DReg>data,
              const std::shared_ptr<SEP::float2DReg>slsqModel,
              float           U_0,
              float 					alpha,
	            int				      spongeWidth);

  void forward(const bool                               add,
               const std::shared_ptr<SEP::complex4DReg> model,
               std::shared_ptr<SEP::complex4DReg>       data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::complex4DReg>      model,
               const std::shared_ptr<SEP::complex4DReg>data) const;
               bool dotTest(const bool verbose = false, const float maxError = .00001) const{
                 std::cerr << "cpp dot test not implemented.\n";
               }
  void set_slsq(std::shared_ptr<SEP::float2DReg>slsq);
private:

  int n1,n2,n3,n4;
  std::shared_ptr<SEP::float2DReg>_slsq;
  std::shared_ptr<SEP::float2DReg>_gamma;
  std::shared_ptr<SEP::float2DReg>_gammaSq;
  std::shared_ptr<SEP::float2DReg>_fatMask;
  int _spongeWidth;
  float _U_0, _alpha;
  float _da, _db, _dw, _ow;
  float C0z, C1z, C2z, C3z, C4z, C5z, C0x, C1x, C2x, C3x, C4x, C5x;

  int FAT = 5;

  float _dt;
};

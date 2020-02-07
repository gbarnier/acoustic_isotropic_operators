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
 #include <complex4DReg.h>

 using namespace SEP;



class WaveRecon_freq_multi_exp_precond : public Operator<SEP::complex4DReg, SEP::complex4DReg> {
public:

  WaveRecon_freq_multi_exp_precond(const std::shared_ptr<SEP::complex4DReg>model,
              const std::shared_ptr<SEP::complex4DReg>data,
              const std::shared_ptr<SEP::float2DReg>slsqModel);

  void forward(const bool                               add,
               const std::shared_ptr<SEP::complex4DReg> model,
               std::shared_ptr<SEP::complex4DReg>       data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::complex4DReg>      model,
               const std::shared_ptr<SEP::complex4DReg>data) const;
   bool dotTest(const bool verbose = false, const float maxError = .00001) const{
     std::cerr << "cpp dot test not implemented.\n";
   }
  void update_slsq(std::shared_ptr<SEP::float2DReg>slsq);
private:

  int n1,n2,n3,n4;
  std::shared_ptr<SEP::float3DReg>_precond;
  float _da, _db, _dw, _ow;
  float C0z, C0x;

  int FAT = 5;

  float _dt;
};

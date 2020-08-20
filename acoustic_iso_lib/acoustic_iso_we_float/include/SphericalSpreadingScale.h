/********************************************//**
   Author: Stuart Farris
   Date: 10JAN2018
   Description:  
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float3DReg.h>
 #include <float1DReg.h>
using namespace SEP;
class SphericalSpreadingScale: public Operator<SEP::float2DReg, SEP::float2DReg> {
public:


  SphericalSpreadingScale(const std::shared_ptr<SEP::float2DReg>model,
                  const std::shared_ptr<SEP::float2DReg>data,
	          const std::shared_ptr<float1DReg> zCoordSou, const std::shared_ptr<float1DReg> xCoordSou,
		  float t_pow, float max_vel);


  void forward(const bool                         add,
               const std::shared_ptr<SEP::float2DReg>model,
               std::shared_ptr<SEP::float2DReg>      data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float2DReg>      model,
               const std::shared_ptr<SEP::float2DReg>data) const;



private:

  std::shared_ptr<float1DReg> _zCoordSou, _xCoordSou; 
  std::shared_ptr<float2DReg> _scale; 

  int _n2;
  int _n1;
  float _o1;
  float _o2;
  float _d1;
  float _d2;
  float _vel;
  float _tpow;

};

class SphericalSpreadingScale_Wfld: public Operator<SEP::float2DReg, SEP::float2DReg> {
public:


  SphericalSpreadingScale_Wfld(const std::shared_ptr<SEP::float2DReg>model,
                  const std::shared_ptr<SEP::float2DReg>data,
	          const std::shared_ptr<float3DReg> wfld);


  void forward(const bool                         add,
               const std::shared_ptr<SEP::float2DReg>model,
               std::shared_ptr<SEP::float2DReg>      data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float2DReg>      model,
               const std::shared_ptr<SEP::float2DReg>data) const;



private:

  std::shared_ptr<float2DReg> _scale; 

  int _n3;
  int _n2;
  int _n1;
  float _o1;
  float _o2;
  float _d1;
  float _d2;

};

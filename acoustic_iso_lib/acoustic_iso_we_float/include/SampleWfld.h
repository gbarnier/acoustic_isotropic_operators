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
class SampleWfld: public Operator<SEP::float3DReg, SEP::float3DReg> {
public:


  SampleWfld(const std::shared_ptr<SEP::float3DReg>model,
                  const std::shared_ptr<SEP::float3DReg>data,
	          const std::shared_ptr<float1DReg> zCoordSou, const std::shared_ptr<float1DReg> xCoordSou,
		  const std::shared_ptr<float1DReg> zCoordRec, const std::shared_ptr<float1DReg> xCoordRec,
		  float t_start, float max_vel);


  void forward(const bool                         add,
               const std::shared_ptr<SEP::float3DReg>model,
               std::shared_ptr<SEP::float3DReg>      data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float3DReg>      model,
               const std::shared_ptr<SEP::float3DReg>data) const;



private:

  std::shared_ptr<float1DReg> _zCoordRec, _xCoordRec; 
  std::shared_ptr<float1DReg> _zCoordSou, _xCoordSou; 
  std::shared_ptr<float3DReg> _mask; 
  float _t_start;
  int _n1;
  int _n2;
  int _n3;
  float _o1;
  float _o2;
  float _o3;
  float _d1;
  float _d2;
  float _d3;
  float _max_vel;

};

class SampleWfldTime: public Operator<SEP::float3DReg, SEP::float3DReg> {
public:


  SampleWfldTime(const std::shared_ptr<SEP::float3DReg>model,
                  const std::shared_ptr<SEP::float3DReg>data,int tmin);


  void forward(const bool                         add,
               const std::shared_ptr<SEP::float3DReg>model,
               std::shared_ptr<SEP::float3DReg>      data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float3DReg>      model,
               const std::shared_ptr<SEP::float3DReg>data) const;



private:

  int   _tmin;
};

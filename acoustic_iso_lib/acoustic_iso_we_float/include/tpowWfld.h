/********************************************//**
   Author: Stuart Farris
   Date: 
   Description:  values in a slice
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float3DReg.h>
using namespace SEP;
class tpowWfld: public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  // regular grid
  tpowWfld(
    const std::shared_ptr<SEP::float3DReg>model,
    const std::shared_ptr<SEP::float3DReg>data,
    float tpow, float tcenter);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float3DReg>model,
                       std::shared_ptr<SEP::float3DReg>      data) const;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float3DReg>      model,
                       const std::shared_ptr<SEP::float3DReg>data) const;

private:

	float _tpow, _tcenter;
	int _n1,_n2,_n3;
	float _d3,_o3;
};

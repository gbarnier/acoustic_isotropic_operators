/********************************************//**
   Author: Stuart Farris
   Date: 
   Description:  Causal mask 
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float3DReg.h>
using namespace SEP;

class CausalMask : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  // regular grid
  CausalMask(
    const std::shared_ptr<SEP::float3DReg>model,
    const std::shared_ptr<SEP::float3DReg>data,
    float vmax,
    float tmin,
    int source_ix,
    int source_iz);

    void forward(const bool                         add,
                 const std::shared_ptr<SEP::float3DReg>model,
                 std::shared_ptr<SEP::float3DReg>      data) const;

    void adjoint(const bool                         add,
                 std::shared_ptr<SEP::float3DReg>      model,
                 const std::shared_ptr<SEP::float3DReg>data) const;

                 /* Destructor */
             		~CausalMask(){};
    
private:

  float _velMax;
  float _lambda = 0.15;
  float _tmin;
  std::shared_ptr<SEP::float3DReg> _mask;
  float _fx_source,_fz_source;
  int _spongeWidth=30;
};


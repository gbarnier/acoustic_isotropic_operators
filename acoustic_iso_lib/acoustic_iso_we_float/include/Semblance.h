/********************************************//**
   Author: Stuart Farris
   Date: 21FEB2018
   Description:  Semblance scan over many cmp gathers. Fwd (tau,s)->(t,off)
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float3DReg.h>
namespace waveform {
class Semblance : public Operator {
public:

  // regular grid
  Semblance(
    const std::shared_ptr<SEP::float3DReg>model,
    const std::shared_ptr<SEP::float3DReg>data)

  virtual void forward(const bool                         add,
                       const std::shared_ptr<SEP::Vector>model,
                       std::shared_ptr<SEP::Vector>data);

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<SEP::Vector>      model,
                       const std::shared_ptr<SEP::Vector>data);

private:

  float _tau0, _t0, _s0, _off0;
  float _dtau, _dt, _ds, _doff;
  int _ntau, _nt, _ns, _noff;
};
}

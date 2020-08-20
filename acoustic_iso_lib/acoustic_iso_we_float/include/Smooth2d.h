/********************************************//**
   Author: Stuart Farris
   Date: 08FEB2018
   Description:  
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float2DReg.h>
using namespace SEP;
class Smooth2d : public Operator<SEP::float2DReg, SEP::float2DReg> {
public:

  // regular grid
  Smooth2d(
    const std::shared_ptr<SEP::float2DReg>model,
    const std::shared_ptr<SEP::float2DReg>data,
    int                                    nfilt1,
    int                                    nfilt2);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float2DReg>model,
                       std::shared_ptr<SEP::float2DReg>      data) const;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float2DReg>      model,
                       const std::shared_ptr<SEP::float2DReg>data) const;

private:

  int _nfilt1,_nfilt2;
  std::shared_ptr<SEP::float2DReg>buffer;
};


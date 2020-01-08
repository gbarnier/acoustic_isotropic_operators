/********************************************//**
   Author: Stuart Farris
   Date: 04JAN2018
   Description:  2nd derivative operator. Deriv taken over fast axis
      3D Vector
 ***********************************************/

#pragma once
#include <operator.h>
#include <float4DReg.h>

using namespace SEP;


class SecondDeriv_multi_exp_V2 : public Operator<SEP::float4DReg, SEP::float4DReg> {
public:

  SecondDeriv_multi_exp_V2(const std::shared_ptr<SEP::float4DReg>model,
              const std::shared_ptr<SEP::float4DReg>data);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float4DReg>model,
                       std::shared_ptr<SEP::float4DReg>      data) const;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float4DReg>      model,
                       const std::shared_ptr<SEP::float4DReg>data) const;

private:

  float C0t_10,C1t_10,C2t_10,C3t_10,C4t_10,C5t_10;
  float C0t_8,C1t_8,C2t_8,C3t_8,C4t_8;
  float C0t_6,C1t_6,C2t_6,C3t_6;
  float C0t_4,C1t_4,C2t_4;
  float C0t_2,C1t_2;
  float _dt; 
  int n1,n2,n3,n4;
};

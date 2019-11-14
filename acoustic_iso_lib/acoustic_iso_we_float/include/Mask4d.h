/********************************************//**
   Author: Stuart Farris
   Date: 08FEB2018
   Description:  Mask values in a 4d float vector
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float4DReg.h>
using namespace SEP;

class Mask4d : public Operator<SEP::float4DReg, SEP::float4DReg> {
public:

  // regular grid
  Mask4d(
    const std::shared_ptr<SEP::float4DReg>model,
    const std::shared_ptr<SEP::float4DReg>data,
    int                                    n1min,
    int                                    n1max,
    int                                    n2min,
    int                                    n2max,
    int                                    n3min,
    int                                    n3max,
    int                                    n4min,
    int                                    n4max,
    int                                    maskType = 0);

    void forward(const bool                         add,
                 const std::shared_ptr<SEP::float4DReg>model,
                 std::shared_ptr<SEP::float4DReg>      data) const;

    void adjoint(const bool                         add,
                 std::shared_ptr<SEP::float4DReg>      model,
                 const std::shared_ptr<SEP::float4DReg>data) const;

                 /* Destructor */
             		~Mask4d(){};

private:

  int _n1,_n2,_n3,_n4;
  int _n1min, _n1max, _n2min, _n2max, _n3min, _n3max, _n4min, _n4max;
  int _maskType;
  std::shared_ptr<float4D>_mask;
};

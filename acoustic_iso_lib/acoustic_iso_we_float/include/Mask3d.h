/********************************************//**
   Author: Stuart Farris
   Date: 08FEB2018
   Description:  Mask values in a cube
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float3DReg.h>
using namespace SEP;

class Mask3d : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:

  // regular grid
  Mask3d(
    const std::shared_ptr<SEP::float3DReg>model,
    const std::shared_ptr<SEP::float3DReg>data,
    int                                    n1min,
    int                                    n1max,
    int                                    n2min,
    int                                    n2max,
    int                                    n3min,
    int                                    n3max,
    int                                    maskType = 0);

    void forward(const bool                         add,
                 const std::shared_ptr<SEP::float3DReg>model,
                 std::shared_ptr<SEP::float3DReg>      data) const;

    void adjoint(const bool                         add,
                 std::shared_ptr<SEP::float3DReg>      model,
                 const std::shared_ptr<SEP::float3DReg>data) const;

                 /* Destructor */
             		~Mask3d(){};

private:

  int _n1min, _n1max, _n2min, _n2max, _n3min, _n3max;
  int _maskType;
  std::shared_ptr<float3D>_mask;
};

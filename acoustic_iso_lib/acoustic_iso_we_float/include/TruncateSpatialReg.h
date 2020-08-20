/********************************************//**
   Author: Stuart Farris
   Date: 10JAN2018
   Description:  Pad and truncate wavefield at regularly sampled spatial
      locations. data[3][2][1] - first dimension is time, second dimension is
      z, third dimenstion is x. The x and z locaitons to pull/push from/to model
      are gathered from data hypercube. This allows the data to be made of
      traces with regular sampling in the model space.

      Truncates/pads around receiver locations within model hypercube. Forward
      3D->3D. Adjoint 3D->3D.
 ***********************************************/
 #pragma once
 #include <operator.h>
 #include <float3DReg.h>
using namespace SEP;
class TruncateSpatialReg : public Operator<SEP::float3DReg, SEP::float3DReg> {
public:


  TruncateSpatialReg(const std::shared_ptr<SEP::float3DReg>model,
                  const std::shared_ptr<SEP::float3DReg>data);


  void forward(const bool                         add,
               const std::shared_ptr<SEP::float3DReg>model,
               std::shared_ptr<SEP::float3DReg>      data) const;

  void adjoint(const bool                         add,
               std::shared_ptr<SEP::float3DReg>      model,
               const std::shared_ptr<SEP::float3DReg>data) const;



private:

  int   n1d,n2d,n3d;
  int   n1m,n2m,n3m;
  float o1d,o2d,o3d;
  float o1m,o2m,o3m;
  float d1d,d2d,d3d;
  float d1m,d2m,d3m;
};

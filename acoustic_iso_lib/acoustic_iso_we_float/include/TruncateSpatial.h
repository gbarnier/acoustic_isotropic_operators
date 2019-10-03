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
 #include <float1DReg.h>
 #include <float2DReg.h>
 #include <float3DReg.h>
using namespace SEP;
class TruncateSpatial : public Operator<SEP::float3DReg, SEP::float2DReg> {
public:

  // regular grid
  // TruncateSpatial(const std::shared_ptr<SEP::float3DReg>model,
  //                 const std::shared_ptr<SEP::float3DReg>data);

  // irregular grid
  // TruncateSpatial(const std::shared_ptr<SEP::float3DReg>model,
  //                 const std::shared_ptr<SEP::float2DReg>data,
  //                 const std::shared_ptr<SEP::float1DReg>xCoordinates,
  //                 const std::shared_ptr<SEP::float1DReg>zCoordinates);
  TruncateSpatial(const std::shared_ptr<SEP::float3DReg>model,
                                   const std::shared_ptr<SEP::float2DReg>data,
                                   const std::shared_ptr<SEP::float2DReg>recCoordinates);

  void forward(const bool                         add,
                       const std::shared_ptr<SEP::float3DReg>model,
                       std::shared_ptr<SEP::float2DReg>      data) const ;

  void adjoint(const bool                         add,
                       std::shared_ptr<SEP::float3DReg>      model,
                       const std::shared_ptr<SEP::float2DReg>data) const ;

  // irregular grid
  void pullToData(const std::shared_ptr<SEP::float3DReg>model,
                  std::shared_ptr<SEP::float2DReg>  data);

  // // regular grid
  // void pullToData(const std::shared_ptr<SEP::Vector>model,
  //                 std::shared_ptr<SEP::float3DReg>  data);

  // irregular grid
  void pushToModel(std::shared_ptr<SEP::float3DReg>          model,
                   const std::shared_ptr<SEP::float2DReg>data);

  // // regular grid
  // void pushToModel(std::shared_ptr<SEP::Vector>          model,
  //                  const std::shared_ptr<SEP::float2DReg>data);

private:

  // std::shared_ptr<SEP::float1DReg>_xCoordinates;
  // std::shared_ptr<SEP::float1DReg>_zCoordinates;
  std::shared_ptr<SEP::float2DReg>_recCoordinates;
};

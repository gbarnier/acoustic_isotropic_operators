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
 #include <Operator.h>
 #include <float1DReg.h>
 #include <float2DReg.h>
 #include <float3DReg.h>
namespace waveform {
class TruncateSpatial : public giee::Operator {
public:

  // regular grid
  TruncateSpatial(const std::shared_ptr<giee::float3DReg>model,
                  const std::shared_ptr<giee::float3DReg>data);

  // irregular grid
  // TruncateSpatial(const std::shared_ptr<giee::float3DReg>model,
  //                 const std::shared_ptr<giee::float2DReg>data,
  //                 const std::shared_ptr<giee::float1DReg>xCoordinates,
  //                 const std::shared_ptr<giee::float1DReg>zCoordinates);
  TruncateSpatial(const std::shared_ptr<giee::float3DReg>model,
                                   const std::shared_ptr<giee::float2DReg>data,
                                   const std::shared_ptr<giee::float2DReg>recCoordinates);    

  virtual void forward(const bool                         add,
                       const std::shared_ptr<giee::Vector>model,
                       std::shared_ptr<giee::Vector>      data);

  virtual void adjoint(const bool                         add,
                       std::shared_ptr<giee::Vector>      model,
                       const std::shared_ptr<giee::Vector>data);

  // irregular grid
  void pullToData(const std::shared_ptr<giee::Vector>model,
                  std::shared_ptr<giee::float2DReg>  data);

  // regular grid
  void pullToData(const std::shared_ptr<giee::Vector>model,
                  std::shared_ptr<giee::float3DReg>  data);

  // irregular grid
  void pushToModel(std::shared_ptr<giee::Vector>          model,
                   const std::shared_ptr<giee::float3DReg>data);

  // regular grid
  void pushToModel(std::shared_ptr<giee::Vector>          model,
                   const std::shared_ptr<giee::float2DReg>data);

private:

  // std::shared_ptr<giee::float1DReg>_xCoordinates;
  // std::shared_ptr<giee::float1DReg>_zCoordinates;
  std::shared_ptr<giee::float2DReg>_recCoordinates;
};
}

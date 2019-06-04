/********************************************//**
 Author: Stuart Farris
 Date: Summer 2017
 Description:  Operator that pads and truncates wavefield at source. Forward 1D->3D. Adjoint 3D->1D. Pads to space discretization. Is not concerned with receiver locations.
 ***********************************************/
#pragma once
//#include </opt/gieeSolver/base/Operator.h>
#include <Operator.h>
#include <float1DReg.h>
#include <float3DReg.h>
namespace waveform{

class PadSource : public giee::Operator {
public:
  /**
  Construct operator. Model will be padded(forward 1D->3D) or truncated(adjoint 3D->1D) around index s2,s3.
  */
  PadSource(const std::shared_ptr<giee::float1DReg> model,
    const std::shared_ptr<giee::float3DReg> data, const int s2, const int s3);

  /**
  Pad (forward 1D->3D) around index s2,s3.
  */
  virtual void forward(const bool add,
    const std::shared_ptr<giee::Vector> model,
    std::shared_ptr<giee::Vector> data);

  /**
  Truncate (forward 1D->3D) around index s2,s3.
  */
  virtual void adjoint(const bool add,
    std::shared_ptr<giee::Vector> model,
    const std::shared_ptr<giee::Vector> data);

private:
  int _s2,_s3; /**< index in second and third dimension of data cube to pad or truncate around */
};
}

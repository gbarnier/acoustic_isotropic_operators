/********************************************//**
 Author: Stuart Farris
 Date: Summer 2017
 Description:  Pad and truncate wavefield at one time slice at receivers. Truncates/pads around receiver locations within model hypercube. Forward 2D->3D. Adjoint 3D->2D.
 ***********************************************/

#pragma once
#include <Operator.h>
#include <float2DReg.h>
#include <float3DReg.h>
namespace waveform{
class PadRec : public giee::Operator {
public:
  /**
  zero pad 2D model to 3D cube around slice s2 in cube. s2 intended to be depth, z. So this pads/truncates all x receiver locations at one time slice. Receiver locations implied to be in model hypercube.
  */
  PadRec(const std::shared_ptr<giee::float2DReg> model,
    const std::shared_ptr<giee::float3DReg> data, const int s2);

  /**
  pad 2D model to 3D cube around slice s2 in cube. s2 intended to be depth, z. So this pads in the z direction and at x locations without receivers. Receiver locations implied to be in model hypercube.
  */
  virtual void forward(const bool add,
    const std::shared_ptr<giee::Vector> model,
    std::shared_ptr<giee::Vector> data);

  /**
  truncate 3D to 2D around slice s2 in cube. s2 intended to be depth, z. So this truncates all z information except in one slice and any x locations that are not receivers. Receiver locations implied to be in model hypercube.
  */
  virtual void adjoint(const bool add,
    std::shared_ptr<giee::Vector> model,
    const std::shared_ptr<giee::Vector> data);

private:
  int _s2; /**< index in second dimension of data cube to pad or truncate around */
  int _nxm, _nxd; /**< number of x locations in model and data, respectively. x is second dimension in model slice but third in data cube */
  float _oxm, _oxd; /**< origin x locations in model and data, respectively. x is second dimension in model slice but third in data cube */
  float _dxm, _dxd; /**< distance between x locations in model and data, respectively. x is second dimension in model slice but third in data cube */
};
}

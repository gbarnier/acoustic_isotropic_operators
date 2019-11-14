#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "WaveReconV2.h"
#include "WaveReconV3.h"
#include "WaveReconV4.h"
#include "WaveReconV5.h"
#include "WaveReconV6.h"
#include "WaveReconV7.h"
#include "WaveReconV8.h"
#include "WaveReconV9.h"
#include "WaveReconV10.h"
#include "PadModel2d.h"
#include "PadModel3d.h"
#include "waveEquationAcousticGpu.h"

namespace py = pybind11;
using namespace SEP;

// Definition of Device object and non-linear propagator
PYBIND11_MODULE(pyAcoustic_iso_float_we, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<PadModel2d, std::shared_ptr<PadModel2d>>(clsGeneric,"PadModel2d")
      .def(py::init<const std::shared_ptr<SEP::float2DReg>, const std::shared_ptr<SEP::float2DReg>, int ,int>(), "Initialize a PadModel2d")

      .def("forward", (void (PadModel2d::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &PadModel2d::forward, "Forward")

      .def("adjoint", (void (PadModel2d::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &PadModel2d::adjoint, "Adjoint")

      .def("dotTest",(bool (PadModel2d::*)(const bool, const float)) &PadModel2d::dotTest,"Dot-Product Test")
  ;

  py::class_<PadModel3d, std::shared_ptr<PadModel3d>>(clsGeneric,"PadModel3d")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>,int,int,int,int>(), "Initialize a PadModel3d")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, int ,int>(), "Initialize a PadModel3d")
      .def("forward", (void (PadModel3d::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &PadModel3d::forward, "Forward")

      .def("adjoint", (void (PadModel3d::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &PadModel3d::adjoint, "Adjoint")

      .def("dotTest",(bool (PadModel3d::*)(const bool, const float)) &PadModel3d::dotTest,"Dot-Product Test")
  ;

  // py::class_<WaveReconV2, std::shared_ptr<WaveReconV2>>(clsGeneric,"WaveReconV2")
  //     .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, int ,int,int,int,int,int,int>(), "Initialize a WaveReconV2")
  //
  //     .def("forward", (void (WaveReconV2::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV2::forward, "Forward")
  //
  //     .def("adjoint", (void (WaveReconV2::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV2::adjoint, "Adjoint")
  //
  //     .def("dotTest",(bool (WaveReconV2::*)(const bool, const float)) &WaveReconV2::dotTest,"Dot-Product Test")
  // ;
  py::class_<WaveReconV3, std::shared_ptr<WaveReconV3>>(clsGeneric,"WaveReconV3")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, int,int>(), "Initialize a WaveReconV3")

      .def("forward", (void (WaveReconV3::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV3::forward, "Forward")

      .def("adjoint", (void (WaveReconV3::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV3::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveReconV3::*)(const bool, const float)) &WaveReconV3::dotTest,"Dot-Product Test")
  ;

  py::class_<WaveReconV4, std::shared_ptr<WaveReconV4>>(clsGeneric,"WaveReconV4")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, int,int>(), "Initialize a WaveReconV4")

      .def("forward", (void (WaveReconV4::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV4::forward, "Forward")

      .def("adjoint", (void (WaveReconV4::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV4::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveReconV4::*)(const bool, const float)) &WaveReconV4::dotTest,"Dot-Product Test")
  ;

  py::class_<WaveReconV5, std::shared_ptr<WaveReconV5>>(clsGeneric,"WaveReconV5")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, int,int>(), "Initialize a WaveReconV5")

      .def("forward", (void (WaveReconV5::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV5::forward, "Forward")

      .def("adjoint", (void (WaveReconV5::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV5::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveReconV5::*)(const bool, const float)) &WaveReconV5::dotTest,"Dot-Product Test")
  ;

  py::class_<WaveReconV6, std::shared_ptr<WaveReconV6>>(clsGeneric,"WaveReconV6")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, int,int>(), "Initialize a WaveReconV6")

      .def("forward", (void (WaveReconV6::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV6::forward, "Forward")

      .def("adjoint", (void (WaveReconV6::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV6::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveReconV6::*)(const bool, const float)) &WaveReconV6::dotTest,"Dot-Product Test")
  ;

  py::class_<WaveReconV7, std::shared_ptr<WaveReconV7>>(clsGeneric,"WaveReconV7")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, int,int>(), "Initialize a WaveReconV7")

      .def("forward", (void (WaveReconV7::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV7::forward, "Forward")

      .def("adjoint", (void (WaveReconV7::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV7::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveReconV7::*)(const bool, const float)) &WaveReconV7::dotTest,"Dot-Product Test")
  ;

  py::class_<WaveReconV8, std::shared_ptr<WaveReconV8>>(clsGeneric,"WaveReconV8")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, float,float,int>(), "Initialize a WaveReconV8")

      .def("forward", (void (WaveReconV8::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV8::forward, "Forward")

      .def("adjoint", (void (WaveReconV8::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV8::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveReconV8::*)(const bool, const float)) &WaveReconV8::dotTest,"Dot-Product Test")
      .def("set_slsq",(void (WaveReconV8::*)(std::shared_ptr<float2DReg>)) &WaveReconV8::set_slsq,"Set slowness squared model")
  ;

  py::class_<WaveReconV9, std::shared_ptr<WaveReconV9>>(clsGeneric,"WaveReconV9")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, float,float,int>(), "Initialize a WaveReconV9")

      .def("forward", (void (WaveReconV9::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV9::forward, "Forward")

      .def("adjoint", (void (WaveReconV9::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV9::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveReconV9::*)(const bool, const float)) &WaveReconV9::dotTest,"Dot-Product Test")
      .def("set_slsq",(void (WaveReconV9::*)(std::shared_ptr<float2DReg>)) &WaveReconV9::set_slsq,"Set slowness squared model")
  ;

  py::class_<WaveReconV10, std::shared_ptr<WaveReconV10>>(clsGeneric,"WaveReconV10")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, float,float,int>(), "Initialize a WaveReconV10")

      .def("forward", (void (WaveReconV10::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV10::forward, "Forward")

      .def("adjoint", (void (WaveReconV10::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV10::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveReconV10::*)(const bool, const float)) &WaveReconV10::dotTest,"Dot-Product Test")
      .def("set_slsq",(void (WaveReconV10::*)(std::shared_ptr<float2DReg>)) &WaveReconV10::set_slsq,"Set slowness squared model")
  ;

  py::class_<WaveRecon_multi_exp, std::shared_ptr<WaveRecon_multi_exp>>(clsGeneric,"WaveRecon_multi_exp")
      .def(py::init<const std::shared_ptr<SEP::float4DReg>, const std::shared_ptr<SEP::float4DReg>, const std::shared_ptr<SEP::float2DReg>, float,float,int>(), "Initialize a WaveRecon_multi_exp")

      .def("forward", (void (WaveRecon_multi_exp::*)(const bool, const std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>)) &WaveRecon_multi_exp::forward, "Forward")

      .def("adjoint", (void (WaveRecon_multi_exp::*)(const bool, const std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>)) &WaveRecon_multi_exp::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveRecon_multi_exp::*)(const bool, const float)) &WaveRecon_multi_exp::dotTest,"Dot-Product Test")
      .def("set_slsq",(void (WaveRecon_multi_exp::*)(std::shared_ptr<float2DReg>)) &WaveRecon_multi_exp::set_slsq,"Set slowness squared model")
  ;

  py::class_<waveEquationAcousticGpu, std::shared_ptr<waveEquationAcousticGpu>>(clsGeneric,"waveEquationAcousticGpu")
      .def(py::init<std::shared_ptr<SEP::float3DReg> , std::shared_ptr<SEP::float3DReg>, std::shared_ptr<SEP::float2DReg>, std::shared_ptr<paramObj>  >(), "Initialize a waveEquationAcousticGpu")

      .def("forward", (void (waveEquationAcousticGpu::*)(const bool, const std::shared_ptr<SEP::float3DReg>, std::shared_ptr<SEP::float3DReg>)) &waveEquationAcousticGpu::forward, "Forward")

      .def("adjoint", (void (waveEquationAcousticGpu::*)(const bool, const std::shared_ptr<SEP::float3DReg>, std::shared_ptr<SEP::float3DReg>)) &waveEquationAcousticGpu::adjoint, "Adjoint")

      .def("dotTest",(bool (waveEquationAcousticGpu::*)(const bool, const float)) &waveEquationAcousticGpu::dotTest,"Dot-Product Test")

      ;
}

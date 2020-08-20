#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "WaveRecon_time.h"
#include "WaveRecon_freq.h"
#include "WaveRecon_freq_precond.h"
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

  py::class_<WaveRecon_time, std::shared_ptr<WaveRecon_time>>(clsGeneric,"WaveRecon_time")
      .def(py::init<const std::shared_ptr<SEP::float4DReg>, const std::shared_ptr<SEP::float4DReg>, const std::shared_ptr<SEP::float2DReg>, float,float,int>(), "Initialize a WaveRecon_time")

      .def("forward", (void (WaveRecon_time::*)(const bool, const std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>)) &WaveRecon_time::forward, "Forward")

      .def("adjoint", (void (WaveRecon_time::*)(const bool, const std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>)) &WaveRecon_time::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveRecon_time::*)(const bool, const float)) &WaveRecon_time::dotTest,"Dot-Product Test")
      .def("set_slsq",(void (WaveRecon_time::*)(std::shared_ptr<float2DReg>)) &WaveRecon_time::set_slsq,"Set slowness squared model")
  ;

  py::class_<waveEquationAcousticGpu, std::shared_ptr<waveEquationAcousticGpu>>(clsGeneric,"waveEquationAcousticGpu")
      .def(py::init<std::shared_ptr<SEP::float3DReg> , std::shared_ptr<SEP::float3DReg>, std::shared_ptr<SEP::float2DReg>, std::shared_ptr<paramObj>  >(), "Initialize a waveEquationAcousticGpu")

      .def("forward", (void (waveEquationAcousticGpu::*)(const bool, const std::shared_ptr<SEP::float3DReg>, std::shared_ptr<SEP::float3DReg>)) &waveEquationAcousticGpu::forward, "Forward")

      .def("adjoint", (void (waveEquationAcousticGpu::*)(const bool, const std::shared_ptr<SEP::float3DReg>, std::shared_ptr<SEP::float3DReg>)) &waveEquationAcousticGpu::adjoint, "Adjoint")

      .def("dotTest",(bool (waveEquationAcousticGpu::*)(const bool, const float)) &waveEquationAcousticGpu::dotTest,"Dot-Product Test")

      ;

  py::class_<WaveRecon_freq, std::shared_ptr<WaveRecon_freq>>(clsGeneric,"WaveRecon_freq")
      .def(py::init<const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::float2DReg>,float>(), "Initialize a WaveRecon_freq")

      .def("forward", (void (WaveRecon_freq::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq::forward, "Forward")

      .def("adjoint", (void (WaveRecon_freq::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveRecon_freq::*)(const bool, const float)) &WaveRecon_freq::dotTest,"Dot-Product Test")
      .def("set_slsq",(void (WaveRecon_freq::*)(std::shared_ptr<float2DReg>)) &WaveRecon_freq::set_slsq,"Set slowness squared model")
  ;

  // py::class_<WaveRecon_freq_precond, std::shared_ptr<WaveRecon_freq_precond>>(clsGeneric,"WaveRecon_freq_precond")
  //     .def(py::init<const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::float2DReg>>(), "Initialize a WaveRecon_freq_precond")
  //
  //     .def("forward", (void (WaveRecon_freq_precond::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq_precond::forward, "Forward")
  //
  //     .def("adjoint", (void (WaveRecon_freq_precond::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq_precond::adjoint, "Adjoint")
  //
  //     .def("dotTest",(bool (WaveRecon_freq_precond::*)(const bool, const float)) &WaveRecon_freq_precond::dotTest,"Dot-Product Test")
  //     .def("update_slsq",(void (WaveRecon_freq_precond::*)(std::shared_ptr<float2DReg>)) &WaveRecon_freq_precond::update_slsq,"Set slowness squared model")
  // ;

}

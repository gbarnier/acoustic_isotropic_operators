#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "WaveReconV2.h"
#include "PadModel2d.h"
#include "PadModel3d.h"

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

  py::class_<WaveReconV2, std::shared_ptr<WaveReconV2>>(clsGeneric,"WaveReconV2")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float2DReg>, int ,int,int,int,int,int,int>(), "Initialize a WaveReconV2")

      .def("forward", (void (WaveReconV2::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV2::forward, "Forward")

      .def("adjoint", (void (WaveReconV2::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &WaveReconV2::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveReconV2::*)(const bool, const float)) &WaveReconV2::dotTest,"Dot-Product Test")
  ;

}

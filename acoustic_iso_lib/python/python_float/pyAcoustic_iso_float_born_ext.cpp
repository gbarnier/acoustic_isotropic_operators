#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "BornExtShotsGpu.h"

namespace py = pybind11;
using namespace SEP;

// Definition of Born operator
PYBIND11_MODULE(pyAcoustic_iso_float_born_ext, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<BornExtShotsGpu, std::shared_ptr<BornExtShotsGpu>>(clsGeneric,"BornExtShotsGpu")
      .def(py::init<std::shared_ptr<SEP::float2DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu>>, std::vector<std::shared_ptr<SEP::float2DReg>>, std::vector<std::shared_ptr<deviceGpu>>>(), "Initialize a BornExtShotsGpu")

      .def("forward", (void (BornExtShotsGpu::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &BornExtShotsGpu::forward, "Forward")

      .def("adjoint", (void (BornExtShotsGpu::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &BornExtShotsGpu::adjoint, "Adjoint")

      .def("forwardWavefield", (void (BornExtShotsGpu::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &BornExtShotsGpu::forwardWavefield, "Forward wavefield")

      .def("adjointWavefield",(void (BornExtShotsGpu::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &BornExtShotsGpu::adjointWavefield, "Adjoint wavefield")

      .def("setVel",(void (BornExtShotsGpu::*)(std::shared_ptr<float2DReg>)) &BornExtShotsGpu::setVel,"Function to set background velocity")

      .def("getVel",(std::shared_ptr<float2DReg> (BornExtShotsGpu::*)()) &BornExtShotsGpu::getVel,"Function to get background velocity")

      .def("dotTest",(bool (BornExtShotsGpu::*)(const bool, const float)) &BornExtShotsGpu::dotTest,"Dot-Product Test")
  ;
}

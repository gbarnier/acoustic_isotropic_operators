#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "tomoExtShotsGpu.h"

namespace py = pybind11;
using namespace SEP;

//Definition of Born operator
PYBIND11_MODULE(pyAcoustic_iso_float_tomo, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<tomoExtShotsGpu, std::shared_ptr<tomoExtShotsGpu>>(clsGeneric,"tomoExtShotsGpu")
      .def(py::init<std::shared_ptr<SEP::float2DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu>>, std::vector<std::shared_ptr<SEP::float2DReg>>, std::vector<std::shared_ptr<deviceGpu>>, std::shared_ptr<SEP::float3DReg>>(), "Initialize a tomoExtShotsGpu")

      .def("forward", (void (tomoExtShotsGpu::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float3DReg>)) &tomoExtShotsGpu::forward, "Forward")

      .def("adjoint", (void (tomoExtShotsGpu::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float3DReg>)) &tomoExtShotsGpu::adjoint, "Adjoint")

      .def("forwardWavefield", (void (tomoExtShotsGpu::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float3DReg>)) &tomoExtShotsGpu::forwardWavefield, "Forward wavefield")

      .def("adjointWavefield",(void (tomoExtShotsGpu::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float3DReg>)) &tomoExtShotsGpu::adjointWavefield, "Adjoint wavefield")

      .def("setVel",(void (tomoExtShotsGpu::*)(std::shared_ptr<float2DReg>)) &tomoExtShotsGpu::setVel,"Function to set background velocity")

      .def("setReflectivityExt",(void (tomoExtShotsGpu::*)(std::shared_ptr<float3DReg>)) &tomoExtShotsGpu::setReflectivityExt,"Function to set reflectivity")

      .def("dotTest",(bool (tomoExtShotsGpu::*)(const bool, const float)) &tomoExtShotsGpu::dotTest,"Dot-Product Test")
  ;
}

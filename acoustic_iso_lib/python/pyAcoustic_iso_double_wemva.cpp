#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "wemvaExtShotsGpu.h"

namespace py = pybind11;
using namespace SEP;

//Definition of Born operator
PYBIND11_MODULE(pyAcoustic_iso_double_wemva, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<wemvaExtShotsGpu, std::shared_ptr<wemvaExtShotsGpu>>(clsGeneric,"wemvaExtShotsGpu")
      .def(py::init<std::shared_ptr<SEP::double2DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu>>, std::vector<std::shared_ptr<SEP::double2DReg>>, std::vector<std::shared_ptr<deviceGpu>>, std::vector<std::shared_ptr<SEP::double2DReg>>>(), "Initialize a wemvaExtShotsGpu")

      .def("forward", (void (wemvaExtShotsGpu::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &wemvaExtShotsGpu::forward, "Forward")

      .def("adjoint", (void (wemvaExtShotsGpu::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &wemvaExtShotsGpu::adjoint, "Adjoint")

      .def("forwardWavefield", (void (wemvaExtShotsGpu::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &wemvaExtShotsGpu::forwardWavefield, "Forward wavefield")

      .def("adjointWavefield",(void (wemvaExtShotsGpu::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &wemvaExtShotsGpu::adjointWavefield, "Adjoint wavefield")

      .def("setVel",(void (wemvaExtShotsGpu::*)(std::shared_ptr<double2DReg>)) &wemvaExtShotsGpu::setVel,"Function to set background velocity")

      .def("dotTest",(bool (wemvaExtShotsGpu::*)(const bool, const float)) &wemvaExtShotsGpu::dotTest,"Dot-Product Test")
  ;
}

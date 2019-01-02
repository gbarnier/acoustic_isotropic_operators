#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "BornShotsGpu.h"

namespace py = pybind11;
using namespace SEP;

//Definition of Born operator
PYBIND11_MODULE(pyAcoustic_iso_double2, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<BornShotsGpu, std::shared_ptr<BornShotsGpu>>(clsGeneric,"BornShotsGpu")
      .def(py::init<std::shared_ptr<SEP::double2DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu>>, std::vector<std::shared_ptr<SEP::double2DReg>>, std::vector<std::shared_ptr<deviceGpu>>>(), "Initialize a BornShotsGpu")

      .def("forward", (void (BornShotsGpu::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &BornShotsGpu::forward, "Forward")

      .def("adjoint", (void (BornShotsGpu::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &BornShotsGpu::adjoint, "Adjoint")

      .def("forwardWavefield", (void (BornShotsGpu::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &BornShotsGpu::forwardWavefield, "Forward wavefield")

      .def("adjointWavefield",(void (BornShotsGpu::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double3DReg>)) &BornShotsGpu::adjointWavefield, "Adjoint wavefield")

      .def("dotTest",(bool (BornShotsGpu::*)(const bool, const float)) &BornShotsGpu::dotTest,"Dot-Product Test")

  ;
}

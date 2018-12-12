#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "nonlinearPropShotsGpu.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyAcoustic_iso_double, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<deviceGpu, std::shared_ptr<deviceGpu>>(clsGeneric, "deviceGpu")
      .def(py::init<const std::shared_ptr<SEP::double1DReg>, const std::shared_ptr<SEP::double1DReg>, const std::shared_ptr<double2DReg>, int &>(), "Initlialize a deviceGPU object using location, velocity, and nt")

      .def(py::init<const std::vector<int> &, const std::vector<int> &, const std::shared_ptr<double2DReg>, int &>(), "Initlialize a deviceGPU object using coordinates and nt")

      .def(py::init<const int &, const int &, const int &, const int &, const int &, const int &, const std::shared_ptr<double2DReg>, int &>(), "Initlialize a deviceGPU object using sampling in z and x axes, velocity, and nt")

      ;

  py::class_<nonlinearPropShotsGpu, std::shared_ptr<nonlinearPropShotsGpu>>(clsGeneric,"nonlinearPropShotsGpu")
      .def(py::init<std::shared_ptr<SEP::double2DReg>, std::shared_ptr<paramObj>, std::vector<std::shared_ptr<deviceGpu>>, std::vector<std::shared_ptr<deviceGpu>>>(), "Initlialize a nonlinearPropShotsGpu")

      .def("forward", (void (nonlinearPropShotsGpu::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu::forward, "Forward")

      .def("adjoint", (void (nonlinearPropShotsGpu::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu::adjoint, "Adjoint")

      .def("forwardWavefield", (void (nonlinearPropShotsGpu::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu::forwardWavefield, "Forward wavefield")

      .def("adjointWavefield",(void (nonlinearPropShotsGpu::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &nonlinearPropShotsGpu::adjointWavefield, "Adjoint wavefield")

      .def("dotTest",(bool (nonlinearPropShotsGpu::*)(const bool, const float)) &nonlinearPropShotsGpu::dotTest,"Dot-Product Test")

      ;
}

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "timeInteg.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyTimeInteg, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<timeInteg, std::shared_ptr<timeInteg>>(clsGeneric,"timeInteg")

      .def(py::init<float>(), "Initialize a timeInteg operator")

      .def("forward", (void (timeInteg::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &timeInteg::forward, "Forward")

      .def("adjoint", (void (timeInteg::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &timeInteg::adjoint, "Adjoint")

  ;

 }

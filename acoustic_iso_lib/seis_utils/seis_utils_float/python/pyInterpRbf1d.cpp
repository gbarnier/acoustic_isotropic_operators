#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "interpRbf1d.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyInterpRbf1d, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<interpRbf1d, std::shared_ptr<interpRbf1d>>(clsGeneric,"interpRbf1d")

      .def(py::init<float,std::shared_ptr<float1DReg>,axis,int,int>(), "Initialize a interpRbf1d operator")

      .def("forward", (void (interpRbf1d::*)(const bool, const std::shared_ptr<float1DReg>, std::shared_ptr<float1DReg>)) &interpRbf1d::forward, "Forward")

      .def("adjoint", (void (interpRbf1d::*)(const bool, const std::shared_ptr<float1DReg>, std::shared_ptr<float1DReg>)) &interpRbf1d::adjoint, "Adjoint")

      .def("getZMesh", (std::shared_ptr<float1DReg> (interpRbf1d::*)()) &interpRbf1d::getZMesh, "getZMesh")

  ;
}

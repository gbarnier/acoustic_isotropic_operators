#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "interpBSpline1d.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyInterpBSpline1d, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<interpBSpline1d, std::shared_ptr<interpBSpline1d>>(clsGeneric,"interpBSpline1d")
      .def(py::init<int,std::shared_ptr<float1DReg>,axis,int,int,float,int>(), "Initialize a interpBSpline1d")

      .def("forward", (void (interpBSpline1d::*)(const bool, const std::shared_ptr<float1DReg>, std::shared_ptr<float1DReg>)) &interpBSpline1d::forward, "Forward")

      .def("adjoint", (void (interpBSpline1d::*)(const bool, const std::shared_ptr<float1DReg>, std::shared_ptr<float1DReg>)) &interpBSpline1d::adjoint, "Adjoint")

      .def("getZMesh", (std::shared_ptr<float1DReg> (interpBSpline1d::*)()) &interpBSpline1d::getZMesh, "getZMesh")

  ;
}

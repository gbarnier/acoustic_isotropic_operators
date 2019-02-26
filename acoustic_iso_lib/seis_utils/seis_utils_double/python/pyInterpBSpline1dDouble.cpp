#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "interpBSpline1dDouble.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyInterpBSpline1dDouble, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<interpBSpline1dDouble, std::shared_ptr<interpBSpline1dDouble>>(clsGeneric,"interpBSpline1dDouble")
      .def(py::init<int,std::shared_ptr<double1DReg>,axis,int,int,double,int>(), "Initialize a interpBSpline1dDouble")

      .def("forward", (void (interpBSpline1dDouble::*)(const bool, const std::shared_ptr<double1DReg>, std::shared_ptr<double1DReg>)) &interpBSpline1dDouble::forward, "Forward")

      .def("adjoint", (void (interpBSpline1dDouble::*)(const bool, const std::shared_ptr<double1DReg>, std::shared_ptr<double1DReg>)) &interpBSpline1dDouble::adjoint, "Adjoint")

  ;
}

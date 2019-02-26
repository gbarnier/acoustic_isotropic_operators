#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "interpBSpline2dDouble.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyInterpBSpline2dDouble, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<interpBSpline2dDouble, std::shared_ptr<interpBSpline2dDouble>>(clsGeneric,"interpBSpline2dDouble")
      .def(py::init<int,int,std::shared_ptr<double1DReg>,std::shared_ptr<double1DReg>,axis,axis,int,int,int,double,double,int>(), "Initialize a interpBSpline2dDouble")

      .def(py::init<int,int,std::shared_ptr<double1DReg>,std::shared_ptr<double1DReg>,axis,axis,std::shared_ptr<double1DReg>,std::shared_ptr<double1DReg>,int,double,double,int>(), "Initialize a interpBSpline2dDouble")

      .def("forward", (void (interpBSpline2dDouble::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double2DReg>)) &interpBSpline2dDouble::forward, "Forward")

      .def("adjoint", (void (interpBSpline2dDouble::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double2DReg>)) &interpBSpline2dDouble::adjoint, "Adjoint")

      .def("getZParamVector", (std::shared_ptr<double1DReg> (interpBSpline2dDouble::*)()) &interpBSpline2dDouble::getZParamVector, "getZParamVector")

      .def("getXParamVector", (std::shared_ptr<double1DReg> (interpBSpline2dDouble::*)()) &interpBSpline2dDouble::getXParamVector, "getXParamVector")
  ;
}

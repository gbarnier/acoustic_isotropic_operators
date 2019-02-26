#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "interpSplineInv.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyInterpSplineInv, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<interpSplineInv, std::shared_ptr<interpSplineInv>>(clsGeneric,"interpSplineInv")
      .def(py::init<axis,axis,std::shared_ptr<double1DReg>,std::shared_ptr<double1DReg>,int,int,int,int,int,double,double,int>(), "Initialize a interpSplineInv")

      .def("forward", (void (interpSplineInv::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double2DReg>)) &interpSplineInv::forward, "Forward")

      .def("adjoint", (void (interpSplineInv::*)(const bool, const std::shared_ptr<double2DReg>, std::shared_ptr<double2DReg>)) &interpSplineInv::adjoint, "Adjoint")

      .def("getZParamVector", (std::shared_ptr<double1DReg> (interpSplineInv::*)()) &interpSplineInv::getZParamVector, "getZParamVector")

      .def("getXParamVector", (std::shared_ptr<double1DReg> (interpSplineInv::*)()) &interpSplineInv::getXParamVector, "getXParamVector")
  ;
}

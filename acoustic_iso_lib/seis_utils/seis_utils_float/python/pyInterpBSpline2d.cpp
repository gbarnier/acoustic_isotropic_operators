#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "interpBSpline2d.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyInterpBSpline2d, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<interpBSpline2d, std::shared_ptr<interpBSpline2d>>(clsGeneric,"interpBSpline2d")
      .def(py::init<int,int,std::shared_ptr<float1DReg>,std::shared_ptr<float1DReg>,axis,axis,int,int,int,float,float,int>(), "Initialize a interpBSpline2d")

      .def(py::init<int,int,std::shared_ptr<float1DReg>,std::shared_ptr<float1DReg>,axis,axis,std::shared_ptr<float1DReg>,std::shared_ptr<float1DReg>,int,float,float,int>(), "Initialize a interpBSpline2d")

      .def("forward", (void (interpBSpline2d::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &interpBSpline2d::forward, "Forward")

      .def("adjoint", (void (interpBSpline2d::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &interpBSpline2d::adjoint, "Adjoint")

      .def("getZParamVector", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getZParamVector, "getZParamVector")

      .def("getXParamVector", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getXParamVector, "getXParamVector")

      .def("getZMesh", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getZMesh, "getZMesh")

      .def("getXMesh", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getXMesh, "getXMesh")

  ;
}

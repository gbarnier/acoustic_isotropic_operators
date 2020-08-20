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

      .def("forward", (void (interpBSpline2d::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &interpBSpline2d::forward, "Forward")

      .def("adjoint", (void (interpBSpline2d::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &interpBSpline2d::adjoint, "Adjoint")

      .def("getZMeshModel", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getZMeshModel, "getZMeshModel")

      .def("getXMeshModel", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getXMeshModel, "getXMeshModel")

      .def("getZMeshData", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getZMeshData, "getZMeshData")

      .def("getXMeshData", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getXMeshData, "getXMeshData")

      .def("getZControlPoints", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getZControlPoints, "getZControlPoints")

      .def("getXControlPoints", (std::shared_ptr<float1DReg> (interpBSpline2d::*)()) &interpBSpline2d::getXControlPoints, "getXControlPoints")

  ;
}

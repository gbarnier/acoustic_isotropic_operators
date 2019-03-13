#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "interpBSpline3d.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyInterpBSpline3d, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<interpBSpline3d, std::shared_ptr<interpBSpline3d>>(clsGeneric,"interpBSpline3d")
      .def(py::init<int,int,int,std::shared_ptr<float1DReg>,std::shared_ptr<float1DReg>,std::shared_ptr<float1DReg>,axis,axis,axis,int,int,int,int,float,float,float,int,int,int>(), "Initialize a interpBSpline3d")

      .def("forward", (void (interpBSpline3d::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &interpBSpline3d::forward, "Forward")

      .def("adjoint", (void (interpBSpline3d::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &interpBSpline3d::adjoint, "Adjoint")

      .def("getZMeshModel", (std::shared_ptr<float1DReg> (interpBSpline3d::*)()) &interpBSpline3d::getZMeshModel, "getZMeshModel")

      .def("getXMeshModel", (std::shared_ptr<float1DReg> (interpBSpline3d::*)()) &interpBSpline3d::getXMeshModel, "getXMeshModel")

      .def("getYMeshModel", (std::shared_ptr<float1DReg> (interpBSpline3d::*)()) &interpBSpline3d::getYMeshModel, "getYMeshModel")

      .def("getZMeshData", (std::shared_ptr<float1DReg> (interpBSpline3d::*)()) &interpBSpline3d::getZMeshData, "getZMeshData")

      .def("getXMeshData", (std::shared_ptr<float1DReg> (interpBSpline3d::*)()) &interpBSpline3d::getXMeshData, "getXMeshData")

      .def("getYMeshData", (std::shared_ptr<float1DReg> (interpBSpline3d::*)()) &interpBSpline3d::getYMeshData, "getYMeshData")

      .def("getZControlPoints", (std::shared_ptr<float1DReg> (interpBSpline3d::*)()) &interpBSpline3d::getZControlPoints, "getZControlPoints")

      .def("getXControlPoints", (std::shared_ptr<float1DReg> (interpBSpline3d::*)()) &interpBSpline3d::getXControlPoints, "getXControlPoints")

      .def("getYControlPoints", (std::shared_ptr<float1DReg> (interpBSpline3d::*)()) &interpBSpline3d::getYControlPoints, "getYControlPoints")

  ;
}

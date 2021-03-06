#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "spatialDeriv.h"
#include "SymesZGrad.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pySpatialDeriv, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<zGrad, std::shared_ptr<zGrad>>(clsGeneric,"zGrad")

      .def(py::init<int>(), "Initialize a z-gradient operator")

      .def("forward", (void (zGrad::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &zGrad::forward, "Forward")

      .def("adjoint", (void (zGrad::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &zGrad::adjoint, "Adjoint")

  ;

  py::class_<xGrad, std::shared_ptr<xGrad>>(clsGeneric,"xGrad")

      .def(py::init<int>(), "Initialize a x-gradient operator")

      .def("forward", (void (xGrad::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &xGrad::forward, "Forward")

      .def("adjoint", (void (xGrad::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &xGrad::adjoint, "Adjoint")

  ;

  py::class_<zxGrad, std::shared_ptr<zxGrad>>(clsGeneric,"zxGrad")

      .def(py::init<int>(), "Initialize a zx-gradient operator")

      .def("forward", (void (zxGrad::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &zxGrad::forward, "Forward")

      .def("adjoint", (void (zxGrad::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &zxGrad::adjoint, "Adjoint")

  ;

  py::class_<Laplacian, std::shared_ptr<Laplacian>>(clsGeneric,"Laplacian")

      .def(py::init<int>(), "Initialize a Laplacian operator")

      .def("forward", (void (Laplacian::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &Laplacian::forward, "Forward")

      .def("adjoint", (void (Laplacian::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &Laplacian::adjoint, "Adjoint")

  ;

  py::class_<SymesZGrad, std::shared_ptr<SymesZGrad>>(clsGeneric,"SymesZGrad")

      .def(py::init<int>(), "Initialize a z-gradient operator for extended images")

      .def("forward", (void (SymesZGrad::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &SymesZGrad::forward, "Forward")

  ;

}

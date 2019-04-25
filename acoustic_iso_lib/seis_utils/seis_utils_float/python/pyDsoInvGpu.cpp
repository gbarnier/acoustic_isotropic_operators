#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "dsoInvGpu.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyDsoInvGpu, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<dsoInvGpu, std::shared_ptr<dsoInvGpu>>(clsGeneric,"dsoInvGpu")

      .def(py::init<int,int,int,int,float>(), "Initialize a DSO Inverse operator")

      .def("forward", (void (dsoInvGpu::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dsoInvGpu::forward, "Forward")

      .def("adjoint", (void (dsoInvGpu::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dsoInvGpu::adjoint, "Adjoint")

  ;
}

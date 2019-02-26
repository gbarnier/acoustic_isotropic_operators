#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "dsoGpu.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyDsoGpu, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<dsoGpu, std::shared_ptr<dsoGpu>>(clsGeneric,"dsoGpu")

      .def(py::init<int,int,int,int,float>(), "Initialize a DSO operator")

      .def("forward", (void (dsoGpu::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dsoGpu::forward, "Forward")

      .def("adjoint", (void (dsoGpu::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dsoGpu::adjoint, "Adjoint")

  ;
}

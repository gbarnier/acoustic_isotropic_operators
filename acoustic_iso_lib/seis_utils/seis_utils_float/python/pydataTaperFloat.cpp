#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "dataTaperFloat.h"

namespace py = pybind11;
using namespace SEP;

// Definition of dataTaperFloat operator
PYBIND11_MODULE(pydataTaperFloat, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<dataTaperFloat, std::shared_ptr<dataTaperFloat>>(clsGeneric,"dataTaperFloat")

      .def(py::init<float,float,float,float,std::shared_ptr<SEP::hypercube>,std::string>(), "Initialize a dataTaper for time muting")

      .def(py::init<float,float,float,std::shared_ptr<SEP::hypercube>>(), "Initialize a dataTaper for offset muting")

      .def("forward", (void (dataTaperFloat::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dataTaperFloat::forward, "Forward")

      .def("adjoint", (void (dataTaperFloat::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dataTaperFloat::adjoint, "Adjoint")

      .def("getTaperMask", (std::shared_ptr<float3DReg> (dataTaperFloat::*)()) &dataTaperFloat::getTaperMask, "getTaperMask")

  ;
}

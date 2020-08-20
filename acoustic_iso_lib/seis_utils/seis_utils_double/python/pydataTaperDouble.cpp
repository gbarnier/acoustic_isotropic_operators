#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "dataTaperDouble.h"

namespace py = pybind11;
using namespace SEP;

// Definition of dataTaperDouble operator
PYBIND11_MODULE(pydataTaperDouble, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<dataTaperDouble, std::shared_ptr<dataTaperDouble>>(clsGeneric,"dataTaperDouble")
      .def(py::init<double,double,double,std::shared_ptr<SEP::hypercube>,std::string>(), "Initialize a dataTaper")

      .def("forward", (void (dataTaperDouble::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &dataTaperDouble::forward, "Forward")

      .def("adjoint", (void (dataTaperDouble::*)(const bool, const std::shared_ptr<double3DReg>, std::shared_ptr<double3DReg>)) &dataTaperDouble::adjoint, "Adjoint")

      .def("getTaperMask", (std::shared_ptr<double3DReg> (dataTaperDouble::*)()) &dataTaperDouble::getTaperMask, "getTaperMask")

  ;
}

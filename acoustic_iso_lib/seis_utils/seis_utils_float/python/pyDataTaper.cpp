#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
#include "dataTaper.h"

namespace py = pybind11;
using namespace SEP;

PYBIND11_MODULE(pyDataTaper, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<dataTaper, std::shared_ptr<dataTaper>>(clsGeneric,"dataTaper")

      .def(py::init<float,float,float,float,std::string,int,float,float,float,int,std::shared_ptr<SEP::hypercube>>(), "Initialize a dataTaper for time and offset muting")

      .def(py::init<float,float,float,float,std::shared_ptr<SEP::hypercube>,std::string,int>(), "Initialize a dataTaper for time muting")

      .def(py::init<float,float,float,std::shared_ptr<SEP::hypercube>,int>(), "Initialize a dataTaper for offset muting")

      .def(py::init<std::shared_ptr<SEP::hypercube>>(), "Initialize a dataTaper for no muting")

      .def("forward", (void (dataTaper::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dataTaper::forward, "Forward")

      .def("adjoint", (void (dataTaper::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &dataTaper::adjoint, "Adjoint")

      .def("getTaperMask", (std::shared_ptr<float3DReg> (dataTaper::*)()) &dataTaper::getTaperMask, "getTaperMask")

      .def("getTaperMaskTime", (std::shared_ptr<float3DReg> (dataTaper::*)()) &dataTaper::getTaperMaskTime, "getTaperMaskTime")

      .def("getTaperMaskOffset", (std::shared_ptr<float3DReg> (dataTaper::*)()) &dataTaper::getTaperMaskOffset, "getTaperMaskOffset")

  ;
}

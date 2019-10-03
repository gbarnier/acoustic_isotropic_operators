/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "GF.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyGF, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<GF, std::shared_ptr<GF>>(clsGeneric,"GF")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>,std::shared_ptr<SEP::float1DReg>,std::shared_ptr<SEP::float1DReg>,float, float>(), "Initialize a GF")

      .def("forward", (void (GF::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &GF::forward, "Forward")

      .def("adjoint", (void (GF::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &GF::adjoint, "Adjoint")

      .def("dotTest",(bool (GF::*)(const bool, const float)) &GF::dotTest,"Dot-Product Test")
  ;
}


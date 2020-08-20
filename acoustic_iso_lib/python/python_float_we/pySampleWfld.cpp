/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "SampleWfld.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pySampleWfld, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<SampleWfld, std::shared_ptr<SampleWfld>>(clsGeneric,"SampleWfld")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>,std::shared_ptr<SEP::float1DReg>,std::shared_ptr<SEP::float1DReg>,std::shared_ptr<SEP::float1DReg>,std::shared_ptr<SEP::float1DReg>,float, float>(), "Initialize a SampleWfld")

      .def("forward", (void (SampleWfld::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &SampleWfld::forward, "Forward")

      .def("adjoint", (void (SampleWfld::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &SampleWfld::adjoint, "Adjoint")

      .def("dotTest",(bool (SampleWfld::*)(const bool, const float)) &SampleWfld::dotTest,"Dot-Product Test")
  ;

  py::class_<SampleWfldTime, std::shared_ptr<SampleWfldTime>>(clsGeneric,"SampleWfldTime")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>,int>(), "Initialize a SampleWfldTime")

      .def("forward", (void (SampleWfldTime::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &SampleWfldTime::forward, "Forward")

      .def("adjoint", (void (SampleWfldTime::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &SampleWfldTime::adjoint, "Adjoint")

      .def("dotTest",(bool (SampleWfldTime::*)(const bool, const float)) &SampleWfldTime::dotTest,"Dot-Product Test")
  ;
}


/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "SphericalSpreadingScale.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pySphericalSpreadingScale, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<SphericalSpreadingScale, std::shared_ptr<SphericalSpreadingScale>>(clsGeneric,"SphericalSpreadingScale")
      .def(py::init<const std::shared_ptr<SEP::float2DReg>, const std::shared_ptr<SEP::float2DReg>,std::shared_ptr<SEP::float1DReg>,std::shared_ptr<SEP::float1DReg>,float, float>(), "Initialize a SphericalSpreadingScale")

      .def("forward", (void (SphericalSpreadingScale::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &SphericalSpreadingScale::forward, "Forward")

      .def("adjoint", (void (SphericalSpreadingScale::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &SphericalSpreadingScale::adjoint, "Adjoint")

      .def("dotTest",(bool (SphericalSpreadingScale::*)(const bool, const float)) &SphericalSpreadingScale::dotTest,"Dot-Product Test")
  ;

  py::class_<SphericalSpreadingScale_Wfld, std::shared_ptr<SphericalSpreadingScale_Wfld>>(clsGeneric,"SphericalSpreadingScale_Wfld")
      .def(py::init<const std::shared_ptr<SEP::float2DReg>, const std::shared_ptr<SEP::float2DReg>,std::shared_ptr<SEP::float3DReg>>(), "Initialize a SphericalSpreadingScale_Wfld")

      .def("forward", (void (SphericalSpreadingScale_Wfld::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &SphericalSpreadingScale_Wfld::forward, "Forward")

      .def("adjoint", (void (SphericalSpreadingScale_Wfld::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &SphericalSpreadingScale_Wfld::adjoint, "Adjoint")

      .def("dotTest",(bool (SphericalSpreadingScale_Wfld::*)(const bool, const float)) &SphericalSpreadingScale_Wfld::dotTest,"Dot-Product Test")
  ;
}


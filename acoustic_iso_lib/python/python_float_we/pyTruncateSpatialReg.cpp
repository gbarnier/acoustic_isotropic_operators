/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "TruncateSpatialReg.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyTruncateSpatialReg, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<TruncateSpatialReg, std::shared_ptr<TruncateSpatialReg>>(clsGeneric,"TruncateSpatialReg")
      .def(py::init<const std::shared_ptr<SEP::float3DReg>, const std::shared_ptr<SEP::float3DReg>>(), "Initialize a TruncateSpatialReg")

      .def("forward", (void (TruncateSpatialReg::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &TruncateSpatialReg::forward, "Forward")

      .def("adjoint", (void (TruncateSpatialReg::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &TruncateSpatialReg::adjoint, "Adjoint")

      .def("dotTest",(bool (TruncateSpatialReg::*)(const bool, const float)) &TruncateSpatialReg::dotTest,"Dot-Product Test")
  ;

}

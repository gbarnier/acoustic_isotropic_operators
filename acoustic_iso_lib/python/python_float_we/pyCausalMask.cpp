/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "CausalMask.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyCausalMask, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<CausalMask, std::shared_ptr<CausalMask>>(clsGeneric,"CausalMask")  //
      .def(py::init<const std::shared_ptr<float3DReg>, const std::shared_ptr<float3DReg>, float, float, int, int>(),"Initlialize CausalMask")

      .def("forward",(void (CausalMask::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &CausalMask::forward,"Forward")

      .def("adjoint",(void (CausalMask::*)(const bool, std::shared_ptr<float3DReg>, const std::shared_ptr<float3DReg>)) &CausalMask::adjoint,"Adjoint")

      .def("dotTest",(bool (CausalMask::*)(const bool, const float)) &CausalMask::dotTest,"Dot-Product Test")

    ;

}



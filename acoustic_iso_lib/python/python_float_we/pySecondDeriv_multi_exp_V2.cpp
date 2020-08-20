/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "SecondDeriv_multi_exp_V2.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pySecondDeriv_multi_exp_V2, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<SecondDeriv_multi_exp_V2, std::shared_ptr<SecondDeriv_multi_exp_V2>>(clsGeneric,"SecondDeriv_multi_exp_V2")  //
      .def(py::init<const std::shared_ptr<float4DReg>, const std::shared_ptr<float4DReg>>(),"Initlialize SecondDeriv_multi_exp_V2")

      .def("forward",(void (SecondDeriv_multi_exp_V2::*)(const bool, const std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>)) &SecondDeriv_multi_exp_V2::forward,"Forward")

      .def("adjoint",(void (SecondDeriv_multi_exp_V2::*)(const bool, std::shared_ptr<float4DReg>, const std::shared_ptr<float4DReg>)) &SecondDeriv_multi_exp_V2::adjoint,"Adjoint")

      .def("dotTest",(bool (SecondDeriv_multi_exp_V2::*)(const bool, const float)) &SecondDeriv_multi_exp_V2::dotTest,"Dot-Product Test")

    ;

}

/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "SecondDeriv_multi_exp_freq.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pySecondDeriv_multi_exp_freq, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<SecondDeriv_multi_exp_freq, std::shared_ptr<SecondDeriv_multi_exp_freq>>(clsGeneric,"SecondDeriv_multi_exp_freq")  //
      .def(py::init<const std::shared_ptr<complex4DReg>, const std::shared_ptr<complex4DReg>>(),"Initlialize SecondDeriv_multi_exp_freq")

      .def("forward",(void (SecondDeriv_multi_exp_freq::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &SecondDeriv_multi_exp_freq::forward,"Forward")

      .def("adjoint",(void (SecondDeriv_multi_exp_freq::*)(const bool, std::shared_ptr<complex4DReg>, const std::shared_ptr<complex4DReg>)) &SecondDeriv_multi_exp_freq::adjoint,"Adjoint")

      .def("dotTest",(bool (SecondDeriv_multi_exp_freq::*)(const bool, const float)) &SecondDeriv_multi_exp_freq::dotTest,"Dot-Product Test")

    ;

}

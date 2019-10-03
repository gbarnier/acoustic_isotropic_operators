/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "Mask2d.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyMask2d, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<Mask2d, std::shared_ptr<Mask2d>>(clsGeneric,"Mask2d")  //
      .def(py::init<const std::shared_ptr<float2DReg>, const std::shared_ptr<float2DReg>, int, int, int, int, int>(),"Initlialize Mask2d")

      .def("forward",(void (Mask2d::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &Mask2d::forward,"Forward")

      .def("adjoint",(void (Mask2d::*)(const bool, std::shared_ptr<float2DReg>, const std::shared_ptr<float2DReg>)) &Mask2d::adjoint,"Adjoint")

      .def("dotTest",(bool (Mask2d::*)(const bool, const float)) &Mask2d::dotTest,"Dot-Product Test")

    ;

}



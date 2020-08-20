/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "Smooth2d.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pySmooth2d, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<Smooth2d, std::shared_ptr<Smooth2d>>(clsGeneric,"Smooth2d")  //
      .def(py::init<const std::shared_ptr<float2DReg>, const std::shared_ptr<float2DReg>, int, int>(),"Initlialize Smooth2d")

      .def("forward",(void (Smooth2d::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &Smooth2d::forward,"Forward")

      .def("adjoint",(void (Smooth2d::*)(const bool, std::shared_ptr<float2DReg>, const std::shared_ptr<float2DReg>)) &Smooth2d::adjoint,"Adjoint")

      .def("dotTest",(bool (Smooth2d::*)(const bool, const float)) &Smooth2d::dotTest,"Dot-Product Test")

    ;

}




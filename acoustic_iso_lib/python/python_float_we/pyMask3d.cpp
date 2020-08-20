/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "Mask3d.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyMask3d, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<Mask3d, std::shared_ptr<Mask3d>>(clsGeneric,"Mask3d")  //
      .def(py::init<const std::shared_ptr<float3DReg>, const std::shared_ptr<float3DReg>, int, int, int, int, int, int, int>(),"Initlialize Mask3d")

      .def("forward",(void (Mask3d::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &Mask3d::forward,"Forward")

      .def("adjoint",(void (Mask3d::*)(const bool, std::shared_ptr<float3DReg>, const std::shared_ptr<float3DReg>)) &Mask3d::adjoint,"Adjoint")

      .def("dotTest",(bool (Mask3d::*)(const bool, const float)) &Mask3d::dotTest,"Dot-Product Test")

    ;

}


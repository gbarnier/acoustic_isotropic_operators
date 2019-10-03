/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "tpowWfld.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyTpowWfld, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<tpowWfld, std::shared_ptr<tpowWfld>>(clsGeneric,"tpowWfld")  //
      .def(py::init<const std::shared_ptr<float3DReg>, const std::shared_ptr<float3DReg>, float,float>(),"Initlialize tpowWfld")

      .def("forward",(void (tpowWfld::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &tpowWfld::forward,"Forward")

      .def("adjoint",(void (tpowWfld::*)(const bool, std::shared_ptr<float3DReg>, const std::shared_ptr<float3DReg>)) &tpowWfld::adjoint,"Adjoint")

      .def("dotTest",(bool (tpowWfld::*)(const bool, const float)) &tpowWfld::dotTest,"Dot-Product Test")

    ;

}

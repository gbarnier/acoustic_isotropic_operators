/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "Mask4d.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyMask4d, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<Mask4d, std::shared_ptr<Mask4d>>(clsGeneric,"Mask4d")  //
      .def(py::init<const std::shared_ptr<float4DReg>, const std::shared_ptr<float4DReg>, int, int, int, int, int, int, int, int, int>(),"Initlialize Mask4d")

      .def("forward",(void (Mask4d::*)(const bool, const std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>)) &Mask4d::forward,"Forward")

      .def("adjoint",(void (Mask4d::*)(const bool, std::shared_ptr<float4DReg>, const std::shared_ptr<float4DReg>)) &Mask4d::adjoint,"Adjoint")

      .def("dotTest",(bool (Mask4d::*)(const bool, const float)) &Mask4d::dotTest,"Dot-Product Test")

    ;

	py::class_<Mask4d_complex, std::shared_ptr<Mask4d_complex>>(clsGeneric,"Mask4d_complex")  //
		.def(py::init<const std::shared_ptr<complex4DReg>, const std::shared_ptr<complex4DReg>, int, int, int, int, int, int, int, int, int>(),"Initlialize Mask4d_complex")

		.def("forward",(void (Mask4d_complex::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &Mask4d_complex::forward,"Forward")

		.def("adjoint",(void (Mask4d_complex::*)(const bool, std::shared_ptr<complex4DReg>, const std::shared_ptr<complex4DReg>)) &Mask4d_complex::adjoint,"Adjoint")

		.def("dotTest",(bool (Mask4d_complex::*)(const bool, const float)) &Mask4d_complex::dotTest,"Dot-Product Test")

	;
}

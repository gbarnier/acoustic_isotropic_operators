/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "Laplacian2d.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyLaplacian2d, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<Laplacian2d, std::shared_ptr<Laplacian2d>>(clsGeneric,"Laplacian2d")  //
      .def(py::init<const std::shared_ptr<float3DReg>, const std::shared_ptr<float3DReg>>(),"Initlialize Laplacian2d")

      .def("forward",(void (Laplacian2d::*)(const bool, const std::shared_ptr<float3DReg>, std::shared_ptr<float3DReg>)) &Laplacian2d::forward,"Forward")

      .def("adjoint",(void (Laplacian2d::*)(const bool, std::shared_ptr<float3DReg>, const std::shared_ptr<float3DReg>)) &Laplacian2d::adjoint,"Adjoint")

      .def("dotTest",(bool (Laplacian2d::*)(const bool, const float)) &Laplacian2d::dotTest,"Dot-Product Test")

    ;

		py::class_<Laplacian2d_multi_exp, std::shared_ptr<Laplacian2d_multi_exp>>(clsGeneric,"Laplacian2d_multi_exp")  //
			.def(py::init<const std::shared_ptr<float4DReg>, const std::shared_ptr<float4DReg>>(),"Initlialize Laplacian2d_multi_exp")

			.def("forward",(void (Laplacian2d_multi_exp::*)(const bool, const std::shared_ptr<float4DReg>, std::shared_ptr<float4DReg>)) &Laplacian2d_multi_exp::forward,"Forward")

			.def("adjoint",(void (Laplacian2d_multi_exp::*)(const bool, std::shared_ptr<float4DReg>, const std::shared_ptr<float4DReg>)) &Laplacian2d_multi_exp::adjoint,"Adjoint")

			.def("dotTest",(bool (Laplacian2d_multi_exp::*)(const bool, const float)) &Laplacian2d_multi_exp::dotTest,"Dot-Product Test")

		;

		py::class_<Laplacian2d_multi_exp_complex, std::shared_ptr<Laplacian2d_multi_exp_complex>>(clsGeneric,"Laplacian2d_multi_exp_complex")  //
			.def(py::init<const std::shared_ptr<complex4DReg>, const std::shared_ptr<complex4DReg>>(),"Initlialize Laplacian2d_multi_exp_complex")

			.def("forward",(void (Laplacian2d_multi_exp_complex::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &Laplacian2d_multi_exp_complex::forward,"Forward")

			.def("adjoint",(void (Laplacian2d_multi_exp_complex::*)(const bool, std::shared_ptr<complex4DReg>, const std::shared_ptr<complex4DReg>)) &Laplacian2d_multi_exp_complex::adjoint,"Adjoint")

			.def("dotTest",(bool (Laplacian2d_multi_exp_complex::*)(const bool, const float)) &Laplacian2d_multi_exp_complex::dotTest,"Dot-Product Test")

		;
}

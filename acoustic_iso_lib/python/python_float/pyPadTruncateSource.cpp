/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "padTruncateSource.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pyPadTruncateSource, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<padTruncateSource, std::shared_ptr<padTruncateSource>>(clsGeneric,"padTruncateSource")  //
      .def(py::init<const std::shared_ptr<float1DReg>, const std::shared_ptr<float1DReg>, const std::shared_ptr<SEP::hypercube>, int&, std::string , int>(),"Initlialize padTruncateSource")

      .def("forward",(void (padTruncateSource::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &padTruncateSource::forward,"Forward")

      .def("adjoint",(void (padTruncateSource::*)(const bool, std::shared_ptr<float2DReg>, const std::shared_ptr<float2DReg>)) &padTruncateSource::adjoint,"Adjoint")

      .def("dotTest",(bool (padTruncateSource::*)(const bool, const float)) &padTruncateSource::dotTest,"Dot-Product Test")

    ;

}

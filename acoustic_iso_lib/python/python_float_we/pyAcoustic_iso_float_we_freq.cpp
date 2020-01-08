#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "WaveRecon_freq.h"

namespace py = pybind11;
using namespace SEP;

// Definition of Device object and non-linear propagator
PYBIND11_MODULE(pyAcoustic_iso_float_we_freq, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<WaveRecon_freq_multi_exp, std::shared_ptr<WaveRecon_freq_multi_exp>>(clsGeneric,"WaveRecon_freq_multi_exp")
      .def(py::init<const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::float2DReg>, float,float,int>(), "Initialize a WaveRecon_freq_multi_exp")

      .def("forward", (void (WaveRecon_freq_multi_exp::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq_multi_exp::forward, "Forward")

      .def("adjoint", (void (WaveRecon_freq_multi_exp::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq_multi_exp::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveRecon_freq_multi_exp::*)(const bool, const float)) &WaveRecon_freq_multi_exp::dotTest,"Dot-Product Test")
      .def("set_slsq",(void (WaveRecon_freq_multi_exp::*)(std::shared_ptr<float2DReg>)) &WaveRecon_freq_multi_exp::set_slsq,"Set slowness squared model")
  ;

}

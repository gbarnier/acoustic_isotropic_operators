#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include "WaveRecon_freq.h"
#include "WaveRecon_freq_V2.h"
#include "WaveRecon_freq_precond.h"

namespace py = pybind11;
using namespace SEP;

// Definition of Device object and non-linear propagator
PYBIND11_MODULE(pyAcoustic_iso_float_we_freq, clsGeneric) {

  py::add_ostream_redirect(clsGeneric, "ostream_redirect");

  py::class_<WaveRecon_freq, std::shared_ptr<WaveRecon_freq>>(clsGeneric,"WaveRecon_freq")
      .def(py::init<const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::float2DReg>,float>(), "Initialize a WaveRecon_freq")

      .def("forward", (void (WaveRecon_freq::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq::forward, "Forward")

      .def("adjoint", (void (WaveRecon_freq::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveRecon_freq::*)(const bool, const float)) &WaveRecon_freq::dotTest,"Dot-Product Test")
      .def("set_slsq",(void (WaveRecon_freq::*)(std::shared_ptr<float2DReg>)) &WaveRecon_freq::set_slsq,"Set slowness squared model")
  ;

  py::class_<WaveRecon_freq_precond, std::shared_ptr<WaveRecon_freq_precond>>(clsGeneric,"WaveRecon_freq_precond")
      .def(py::init<const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::float2DReg>>(), "Initialize a WaveRecon_freq_precond")

      .def("forward", (void (WaveRecon_freq_precond::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq_precond::forward, "Forward")

      .def("adjoint", (void (WaveRecon_freq_precond::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq_precond::adjoint, "Adjoint")

      .def("dotTest",(bool (WaveRecon_freq_precond::*)(const bool, const float)) &WaveRecon_freq_precond::dotTest,"Dot-Product Test")
      .def("update_slsq",(void (WaveRecon_freq_precond::*)(std::shared_ptr<float2DReg>)) &WaveRecon_freq_precond::update_slsq,"Set slowness squared model")
  ;

  // py::class_<WaveRecon_freq_V2, std::shared_ptr<WaveRecon_freq_V2>>(clsGeneric,"WaveRecon_freq_V2")
  //     .def(py::init<const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::complex4DReg>, const std::shared_ptr<SEP::float2DReg>,float>(), "Initialize a WaveRecon_freq_V2")
  //
  //     .def("forward", (void (WaveRecon_freq_V2::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq_V2::forward, "Forward")
  //
  //     .def("adjoint", (void (WaveRecon_freq_V2::*)(const bool, const std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>)) &WaveRecon_freq_V2::adjoint, "Adjoint")
  //
  //     .def("dotTest",(bool (WaveRecon_freq_V2::*)(const bool, const float)) &WaveRecon_freq_V2::dotTest,"Dot-Product Test")
  //     .def("set_slsq",(void (WaveRecon_freq_V2::*)(std::shared_ptr<float2DReg>)) &WaveRecon_freq_V2::set_slsq,"Set slowness squared model")
  // ;

}

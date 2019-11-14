/*PyBind11 header files*/
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>
/*Library header files*/
#include "spaceInterp.h"

namespace py = pybind11;
using namespace SEP;


PYBIND11_MODULE(pySpaceInterpFloat, clsGeneric) {
  //Necessary to redirect std::cout into python stdout
	py::add_ostream_redirect(clsGeneric, "ostream_redirect");

    py::class_<spaceInterp, std::shared_ptr<spaceInterp>>(clsGeneric,"spaceInterp")  //
      .def(py::init<const std::shared_ptr<float1DReg>, const std::shared_ptr<float1DReg>, const std::shared_ptr<SEP::hypercube>, int&, std::string , int>(),"Initlialize spaceInterp")

      .def("forward",(void (spaceInterp::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &spaceInterp::forward,"Forward")

      .def("adjoint",(void (spaceInterp::*)(const bool, std::shared_ptr<float2DReg>, const std::shared_ptr<float2DReg>)) &spaceInterp::adjoint,"Adjoint")

      .def("dotTest",(bool (spaceInterp::*)(const bool, const float)) &spaceInterp::dotTest,"Dot-Product Test")

			.def("getNDeviceReg",(int (spaceInterp::*)())&spaceInterp::getNDeviceReg,"Get number of regular devices")

			.def("getNDeviceIrreg",(int (spaceInterp::*)())&spaceInterp::getNDeviceIrreg,"Get number of regular devices")

			.def("getRegPosUniqueVector",(std::vector<int> (spaceInterp::*)())&spaceInterp::getRegPosUniqueVector,"Get vector of unique grid point locations")
    ;

		py::class_<spaceInterp_multi_exp, std::shared_ptr<spaceInterp_multi_exp>>(clsGeneric,"spaceInterp_multi_exp")  //
			.def(py::init<const std::shared_ptr<float1DReg>, const std::shared_ptr<float1DReg>, const std::shared_ptr<float1DReg>, const std::shared_ptr<SEP::hypercube>, int&, std::string , int>(),"Initlialize spaceInterp_multi_exp")

			.def("forward",(void (spaceInterp_multi_exp::*)(const bool, const std::shared_ptr<float2DReg>, std::shared_ptr<float2DReg>)) &spaceInterp_multi_exp::forward,"Forward")

			.def("adjoint",(void (spaceInterp_multi_exp::*)(const bool, std::shared_ptr<float2DReg>, const std::shared_ptr<float2DReg>)) &spaceInterp_multi_exp::adjoint,"Adjoint")

			.def("dotTest",(bool (spaceInterp_multi_exp::*)(const bool, const float)) &spaceInterp_multi_exp::dotTest,"Dot-Product Test")

			.def("getNDeviceReg",(int (spaceInterp_multi_exp::*)())&spaceInterp_multi_exp::getNDeviceReg,"Get number of regular devices")

			.def("getNDeviceIrreg",(int (spaceInterp_multi_exp::*)())&spaceInterp_multi_exp::getNDeviceIrreg,"Get number of regular devices")

			.def("getRegPosUniqueVector",(std::vector<std::vector<int>> (spaceInterp_multi_exp::*)())&spaceInterp_multi_exp::getRegPosUniqueVector,"Get vector of unique grid point locations")

			.def("getIndexMaps",(std::vector<std::map<int,int>> (spaceInterp_multi_exp::*)())&spaceInterp_multi_exp::getIndexMaps,"Get maps from grid locations to regular source traces")
		;

}

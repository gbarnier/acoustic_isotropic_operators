cmake_minimum_required(VERSION 2.8)
project(pyAcoustic_iso_float)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# include folder for ACOUSTIC_ISO_FLOAT library
#include_directories(../../acoustic_iso_float/include)
include_directories(../../acoustic_iso_we_float/include)

#PYBIND11 library headers
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})
#library headers
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../acoustic_isotropic_operators")

find_package(PythonLibs)
include_directories(${PYTHON_INCLUDE_PATH})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${GENERIC_EXPORT_INC_DIRS})
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
                               ${hypercube_BINARY_DIR}
                               ${buffers-lib_BINARY_DIR}
                               ${json_BINARY_DIR}
                               ${sepVector-lib_BINARY_DIR}
                               ${SEPlib_LIBDIR}
)

set(LIBS genericCpp buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

# Compiling and adding module to library
# pybind11_add_module(pyAcoustic_iso_float_nl MODULE pyAcoustic_iso_float_nl.cpp)
# target_link_libraries(pyAcoustic_iso_float_nl PRIVATE ACOUSTIC_ISO_FLOAT_NL ${LIBS})
# pybind11_add_module(pyAcoustic_iso_float_born MODULE pyAcoustic_iso_float_born.cpp)
# target_link_libraries(pyAcoustic_iso_float_born PRIVATE ACOUSTIC_ISO_FLOAT_BORN ${LIBS})
# pybind11_add_module(pyAcoustic_iso_float_born_ext MODULE pyAcoustic_iso_float_born_ext.cpp)
# target_link_libraries(pyAcoustic_iso_float_born_ext PRIVATE ACOUSTIC_ISO_FLOAT_BORN_EXT ${LIBS})
# pybind11_add_module(pyAcoustic_iso_float_tomo MODULE pyAcoustic_iso_float_tomo.cpp)
# target_link_libraries(pyAcoustic_iso_float_tomo PRIVATE ACOUSTIC_ISO_FLOAT_TOMO_EXT ${LIBS})
# pybind11_add_module(pyAcoustic_iso_float_wemva MODULE pyAcoustic_iso_float_wemva.cpp)
# target_link_libraries(pyAcoustic_iso_float_wemva PRIVATE ACOUSTIC_ISO_FLOAT_WEMVA_EXT ${LIBS})
pybind11_add_module(pyAcoustic_iso_float_we MODULE pyAcoustic_iso_float_we.cpp)
target_link_libraries(pyAcoustic_iso_float_we PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pyTruncateSpatialReg MODULE pyTruncateSpatialReg.cpp)
target_link_libraries(pyTruncateSpatialReg PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pySpaceInterpFloat MODULE pySpaceInterpFloat.cpp)
target_link_libraries(pySpaceInterpFloat PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pyPadTruncateSource MODULE pyPadTruncateSource.cpp)
target_link_libraries(pyPadTruncateSource PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pyTpowWfld MODULE pyTpowWfld.cpp)
target_link_libraries(pyTpowWfld PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pyMask4d MODULE pyMask4d.cpp)
target_link_libraries(pyMask4d PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pyMask3d MODULE pyMask3d.cpp)
target_link_libraries(pyMask3d PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pyMask2d MODULE pyMask2d.cpp)
target_link_libraries(pyMask2d PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pyLaplacian2d MODULE pyLaplacian2d.cpp)
target_link_libraries(pyLaplacian2d PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pyGradio MODULE pyGradio.cpp)
target_link_libraries(pyGradio PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pySampleWfld MODULE pySampleWfld.cpp)
target_link_libraries(pySampleWfld PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pySmooth2d MODULE pySmooth2d.cpp)
target_link_libraries(pySmooth2d PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pyCausalMask MODULE pyCausalMask.cpp)
target_link_libraries(pyCausalMask PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pySphericalSpreadingScale MODULE pySphericalSpreading.cpp)
target_link_libraries(pySphericalSpreadingScale PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pySecondDeriv_V2 MODULE pySecondDeriv_V2.cpp)
target_link_libraries(pySecondDeriv_V2 PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pySecondDeriv_multi_exp_V2 MODULE pySecondDeriv_multi_exp_V2.cpp)
target_link_libraries(pySecondDeriv_multi_exp_V2 PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})
pybind11_add_module(pySecondDeriv_multi_exp_freq MODULE pySecondDeriv_multi_exp_freq.cpp)
target_link_libraries(pySecondDeriv_multi_exp_freq PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})

# Installing library
set(PYTHON_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
# install( TARGETS pyAcoustic_iso_float_nl DESTINATION ${PYTHON_DIR}) # Module for non-linear operator
# install( TARGETS pyAcoustic_iso_float_born DESTINATION ${PYTHON_DIR}) # Module for Born operator
# install( TARGETS pyAcoustic_iso_float_born_ext DESTINATION ${PYTHON_DIR}) # Module for Extended Born operator
# install( TARGETS pyAcoustic_iso_float_tomo DESTINATION ${PYTHON_DIR}) # Module for Extended Tomo operator
# install( TARGETS pyAcoustic_iso_float_wemva DESTINATION ${PYTHON_DIR}) # Module for Extended Wemva operator
install( TARGETS pyAcoustic_iso_float_we DESTINATION ${PYTHON_DIR})
install( TARGETS pyTruncateSpatialReg DESTINATION ${PYTHON_DIR}) #
install( TARGETS pySpaceInterpFloat DESTINATION ${PYTHON_DIR})
install( TARGETS pyPadTruncateSource DESTINATION ${PYTHON_DIR})
install( TARGETS pyTpowWfld DESTINATION ${PYTHON_DIR})
install( TARGETS pyMask4d DESTINATION ${PYTHON_DIR})
install( TARGETS pyMask3d DESTINATION ${PYTHON_DIR})
install( TARGETS pyMask2d DESTINATION ${PYTHON_DIR})
install( TARGETS pyLaplacian2d DESTINATION ${PYTHON_DIR})
install( TARGETS pyGradio DESTINATION ${PYTHON_DIR})
install( TARGETS pySampleWfld DESTINATION ${PYTHON_DIR})
install( TARGETS pySmooth2d DESTINATION ${PYTHON_DIR})
install( TARGETS pyCausalMask DESTINATION ${PYTHON_DIR})
install( TARGETS pySphericalSpreadingScale DESTINATION ${PYTHON_DIR})
install( TARGETS pySecondDeriv_V2 DESTINATION ${PYTHON_DIR})
install( TARGETS pySecondDeriv_multi_exp_V2 DESTINATION ${PYTHON_DIR})
install( TARGETS pySecondDeriv_multi_exp_freq DESTINATION ${PYTHON_DIR})

# install( FILES Acoustic_iso_float.py DESTINATION ${PYTHON_DIR})
install( FILES Acoustic_iso_float_we.py DESTINATION ${PYTHON_DIR})
install( FILES TruncateSpatialReg.py DESTINATION ${PYTHON_DIR})
install( FILES wriUtilFloat.py DESTINATION ${PYTHON_DIR})
install( FILES SpaceInterpFloat.py DESTINATION ${PYTHON_DIR})
install( FILES PadTruncateSource.py DESTINATION ${PYTHON_DIR})
install( FILES TpowWfld.py DESTINATION ${PYTHON_DIR})
install( FILES Laplacian2d.py DESTINATION ${PYTHON_DIR})
install( FILES Mask4d.py DESTINATION ${PYTHON_DIR})
install( FILES Mask3d.py DESTINATION ${PYTHON_DIR})
install( FILES Mask2d.py DESTINATION ${PYTHON_DIR})
install( FILES Acoustic_iso_float_gradio.py DESTINATION ${PYTHON_DIR})
install( FILES SampleWfld.py DESTINATION ${PYTHON_DIR})
install( FILES Smooth2d.py DESTINATION ${PYTHON_DIR})
install( FILES SphericalSpreadingScale.py DESTINATION ${PYTHON_DIR})
install( FILES SecondDeriv.py DESTINATION ${PYTHON_DIR})
install( FILES wri_gradient_script.py DESTINATION ${PYTHON_DIR})
install( FILES fft_wfld.py DESTINATION ${PYTHON_DIR})

# Installing Python executables
# install( FILES nonlinearPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES BornPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES BornExtPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES tomoExtPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES wemvaExtPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES fwiMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES fwimeMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES lsrtmMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES extLsrtmMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES wemvaMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES tomoLinearMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES SymesPseudoInvMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES SymesExtLsrtmMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
# install( FILES fwimeFwiObjMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveEquationPythonAcousticFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveEquationPythonAcousticFloatMain_multi_exp.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveEquationPythonAcousticFloatMain_multi_exp_freq.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES truncateSpatialMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES forcingTermMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveEqInvFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveEqInvFloat_multi_exp.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveEqInvFloat_multi_exp_freq.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveReconMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveReconMainFloat_multi_exp.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveReconMainFloat_multi_exp_freq.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES mask3dMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES GradiometryMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES GradiometryMainFloat_multi_exp.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES GradiometryMainFloat_multi_exp_freq.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES tpowMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES sampleWfldMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES smooth2dMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES laplacianTesting.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES waveEquationSpeedTest.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES causalMaskMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES recSamplingTest.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES sphericalSpreadingScaleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES gfMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES wri_fullVp.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES wri_gradient_script.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES fwi_gradient_script.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES gradEditTest.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES fftWfldMainFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES generateRecParFileMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES secondDerivMainFloat_multi_exp.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES secondDerivMainFloat_multi_exp_freq.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

cmake_minimum_required(VERSION 2.8)
project(ACOUSTIC_ISO_FLOAT-lib)

#Adding include files
include_directories(include)
include_directories(${Boost_INCLUDE_DIR})
set(SRC_DIR src)
set(SOURCES
    ${SRC_DIR}/waveEquationAcousticGpu.cpp
  # ${SRC_DIR}/kernelsGpuWaveEquationAcoustic.cu
    ${SRC_DIR}/waveEquationAcousticGpuFunctions.cu
  # ${SRC_DIR}/cudaError.cu
    ${SRC_DIR}/fdParamAcousticWaveEquation.cpp
  ${SRC_DIR}/Laplacian2d.cpp
  ${SRC_DIR}/TruncateSpatial.cpp
  ${SRC_DIR}/TruncateSpatialReg.cpp
  ${SRC_DIR}/SecondDeriv.cpp
  ${SRC_DIR}/SecondDeriv_V2.cpp
  ${SRC_DIR}/SecondDeriv_multi_exp_V2.cpp
  ${SRC_DIR}/SecondDeriv_multi_exp_freq.cpp
  ${SRC_DIR}/WaveRecon_freq.cpp
  ${SRC_DIR}/WaveRecon_time.cpp
  ${SRC_DIR}/SphericalSpreadingScale.cpp
  ${SRC_DIR}/PadModel2d.cpp
  ${SRC_DIR}/PadModel3d.cpp
  ${SRC_DIR}/Mask3d.cpp
  ${SRC_DIR}/Mask4d.cpp
  ${SRC_DIR}/CausalMask.cpp
  ${SRC_DIR}/spaceInterp.cpp
  ${SRC_DIR}/padTruncateSource.cpp
  ${SRC_DIR}/tpowWfld.cpp
  ${SRC_DIR}/Mask2d.cpp
  ${SRC_DIR}/Gradio.cpp
  ${SRC_DIR}/Gradio_multi_exp.cpp
  ${SRC_DIR}/Gradio_multi_exp_freq.cpp
  ${SRC_DIR}/SampleWfld.cpp
  ${SRC_DIR}/Smooth2d.cpp
  )

set(INC_DIR include)
set(HEADERS
   ${INC_DIR}/waveEquationAcousticGpu.h
   ${SRC_DIR}/kernelsGpuWaveEquationAcoustic.cu
   ${SRC_DIR}/waveEquationAcousticGpuFunctions.cu
   ${SRC_DIR}/cudaErrors.cu
   ${INC_DIR}/varDeclareWaveEquation.h
   ${INC_DIR}/waveEquationAcousticGpuFunctions.h
   ${INC_DIR}/fdParamAcousticWaveEquation.h
  ${INC_DIR}/Laplacian2d.h
  ${INC_DIR}/TruncateSpatial.h
  ${INC_DIR}/TruncateSpatialReg.h
  ${INC_DIR}/SecondDeriv.h
  ${INC_DIR}/SecondDeriv_V2.h
  ${INC_DIR}/SecondDeriv_multi_exp_V2.h
  ${INC_DIR}/SecondDeriv_multi_exp_freq.h
  ${INC_DIR}/WaveRecon_time.h
  ${INC_DIR}/WaveRecon_freq.h
  ${INC_DIR}/WaveRecon_freq_precond.h
  ${INC_DIR}/SphericalSpreadingScale.h
  ${INC_DIR}/PadModel2d.h
  ${INC_DIR}/PadModel3d.h
  ${INC_DIR}/Mask3d.h
  ${INC_DIR}/Mask4d.h
  ${INC_DIR}/CausalMask.h
  ${INC_DIR}/spaceInterp.h
  ${INC_DIR}/padTruncateSource.h
  ${INC_DIR}/tpowWfld.h
  ${INC_DIR}/Mask2d.h
  ${INC_DIR}/Gradio.h
  ${INC_DIR}/Gradio_multi_exp.h
  ${INC_DIR}/Gradio_multi_exp_freq.h
  ${INC_DIR}/SampleWfld.h
  ${INC_DIR}/Smooth2d.h
)

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
				 ${hypercube_BINARY_DIR}
				 ${buffers-lib_BINARY_DIR}
				 ${json_BINARY_DIR}
				 ${sepVector-lib_BINARY_DIR}
				 ${SEPlib_LIBDIR}
)
#Setting io libraries
set(LIBS genericCpp sepVector sep3d sep)


set(LIBS genericCpp sepVector sep3d sep)
add_library(ACOUSTIC_ISO_FLOAT_WE  ${SOURCES})
target_link_libraries(ACOUSTIC_ISO_FLOAT_WE ${LIBS} OpenMP::OpenMP_CXX)
install(FILES ${HEADERS} DESTINATION include)
install(TARGETS ACOUSTIC_ISO_FLOAT_WE DESTINATION lib)

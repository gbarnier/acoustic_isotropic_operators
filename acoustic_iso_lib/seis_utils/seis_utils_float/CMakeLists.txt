cmake_minimum_required(VERSION 2.8)
project(SEIS_UTILS_FLOAT-lib)

# include folder for ACOUSTIC_ISO_DOUBLE library
include_directories(../../acoustic_iso_float/include)

#Adding python programs
add_subdirectory(python)

#Adding include files
include_directories(include)

#Files to be compiled
set(SRC_DIR src)
#Files for nonLinear operator library
set(SOURCE_NL ${SRC_DIR}/dataTaper.cpp
			  ${SRC_DIR}/interpBSpline2d.cpp
			  ${SRC_DIR}/spatialDeriv.cpp
			  ${SRC_DIR}/dsoGpu.cpp
)

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
				 ${hypercube_BINARY_DIR}
				 ${buffers-lib_BINARY_DIR}
				 ${json_BINARY_DIR}
				 ${sepVector-lib_BINARY_DIR}
				 ${SEPlib_LIBDIR}
)

# Compiling the project library
add_library(SEIS_UTILS_FLOAT_LIB ${SOURCE_NL})
# Compiling the project executables (if any)

# B-spline interpolation program
# add_executable(interpSplineTestMain ${SRC_DIR}/interpSplineTestMain.cpp)
# target_link_libraries(interpSplineTestMain genericCpp OpenMP::OpenMP_CXX)

# Data muting program
# add_executable(dataTaperDoubleMain ${SRC_DIR}/dataTaperDoubleMain.cpp)
# target_link_libraries(dataTaperDoubleMain genericCpp OpenMP::OpenMP_CXX)

#Adding libray and header files to install destination
install(TARGETS SEIS_UTILS_FLOAT_LIB DESTINATION lib)
# install(TARGETS interpSplineTestMain DESTINATION bin)
# install(TARGETS dataTaperDoubleMain DESTINATION bin)
# install(FILES ${HEADERS} DESTINATION include)

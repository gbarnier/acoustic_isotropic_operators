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
			  ${SRC_DIR}/interpBSpline1d.cpp
			  ${SRC_DIR}/interpBSpline2d.cpp
			  ${SRC_DIR}/interpBSpline3d.cpp
			  ${SRC_DIR}/spatialDeriv.cpp
			  ${SRC_DIR}/dsoGpu.cpp
			  ${SRC_DIR}/dsoInvGpu.cpp
			  ${SRC_DIR}/interpRbf1d.cpp
			  ${SRC_DIR}/timeInteg.cpp
			  ${SRC_DIR}/SymesZGrad.cpp
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

# Compare data
add_executable(dataCompMain ${SRC_DIR}/dataCompMain.cpp)
target_link_libraries(dataCompMain SEIS_UTILS_FLOAT_LIB genericCpp OpenMP::OpenMP_CXX)

# Build initial model for models with salt
add_executable(saltModelMain ${SRC_DIR}/saltModelMain.cpp)
target_link_libraries(saltModelMain SEIS_UTILS_FLOAT_LIB genericCpp OpenMP::OpenMP_CXX)

# Spatial derivatives
add_executable(zGradJunkMain ${SRC_DIR}/zGradJunkMain.cpp)
target_link_libraries(zGradJunkMain genericCpp OpenMP::OpenMP_CXX)

#Adding libray and header files to install destination
install(TARGETS SEIS_UTILS_FLOAT_LIB DESTINATION lib)
install(TARGETS dataCompMain DESTINATION bin)
install(TARGETS saltModelMain DESTINATION bin)
install(TARGETS zGradJunkMain DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)

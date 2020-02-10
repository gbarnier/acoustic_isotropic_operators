cmake_minimum_required(VERSION 2.8)
project(SEIS_UTILS_DOUBLE-lib)

# include folder for ACOUSTIC_ISO_DOUBLE library
include_directories(../../acoustic_iso_double/include)
include_directories(${Boost_INCLUDE_DIR})

#Adding python programs
add_subdirectory(python)

#Adding include files
include_directories(include)

#Files to be compiled
set(SRC_DIR src)
#Files for nonLinear operator library
set(SOURCE_NL ${SRC_DIR}/dataTaperDouble.cpp
			  ${SRC_DIR}/interpBSpline1dDouble.cpp
			  ${SRC_DIR}/interpBSpline2dDouble.cpp

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

#Compiling the project library
add_library(SEIS_UTILS_DOUBLE_LIB ${SOURCE_NL})
target_link_libraries(SEIS_UTILS_DOUBLE_LIB ${LIBS} OpenMP::OpenMP_CXX)
# Compiling the project executables (if any)

# Data muting program
# add_executable(dataTaperDoubleMain ${SRC_DIR}/dataTaperDoubleMain.cpp)
# target_link_libraries(dataTaperDoubleMain ${LIBS} OpenMP::OpenMP_CXX)

#Adding libray and header files to install destination
install(TARGETS SEIS_UTILS_DOUBLE_LIB DESTINATION lib)
# install(TARGETS dataTaperDoubleMain DESTINATION bin)
# install(FILES ${HEADERS} DESTINATION include)

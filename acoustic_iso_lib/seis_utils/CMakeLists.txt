cmake_minimum_required(VERSION 2.8)
project(SEIS_UTILS-lib)

#Adding python programs
add_subdirectory(python)

#Adding include files
include_directories(include)

#Files to be compiled
set(SRC_DIR src)
#Files for nonLinear operator library
set(SOURCE_NL ${SRC_DIR}/dataTaper.cpp
)

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
				 ${hypercube_BINARY_DIR}
				 ${buffers-lib_BINARY_DIR}
				 ${json_BINARY_DIR}
				 ${sepVector-lib_BINARY_DIR}
				 ${SEPlib_LIBDIR}
)

#Compiling the project library
add_library(SEIS_UTILS_LIB ${SOURCE_NL})
# Compiling the project executables

# Data muting program
add_executable(dataTaperMain ${SRC_DIR}/dataTaperMain.cpp)
target_link_libraries(dataTaperMain genericCpp OpenMP::OpenMP_CXX)

#Adding libray and header files to install destination
install(TARGETS SEIS_UTILS_LIB DESTINATION lib)
install(TARGETS dataTaperMain DESTINATION bin)
# install(FILES ${HEADERS} DESTINATION include)

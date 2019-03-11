cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(INSTALL_CMAKE_ENV OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
message("${CMAKE_MODULE_PATH} CMAKE PATH")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE True)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

#Setting project name and required languages
project(ACOUSTIC_ISOTROPIC_OPERATORS LANGUAGES CXX CUDA)
#Setting version
set(ACOUSTIC_ISOTROPIC_OPERATORS_MAJOR_VERSION 0)
set(ACOUSTIC_ISOTROPIC_OPERATORS_MINOR_VERSION 1)
set(ACOUSTIC_ISOTROPIC_OPERATORS_STRING ${ACOUSTIC_ISOTROPIC_OPERATORS_MAJOR_VERSION}.${ACOUSTIC_ISOTROPIC_OPERATORS_MINOR_VERSION})
#Compiler flags
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3")

#Required packages
find_package(Boost REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#Adding genericIO submodule
# execute_process(COMMAND git submodule add http://cees-gitlab.Stanford.EDU/bob/genericIO.git external/genericIO WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#Updating genericIO library
  #execute_process(COMMAND git submodule update --init --recursive --remote external/genericIO WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(external/genericIO)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/external/genericIO)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/external/genericIO/external/sepVector)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/cpp)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/sepVector/lib)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/sepVector/external/hypercube/lib)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/buffers/external/json)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/buffers/lib)


# Finding multithread libraries
find_package(Threads)
# Adding TBB library
find_package(TBB REQUIRED)

# Acoustic isotropic operators (Double precision)
add_subdirectory(acoustic_iso_double)
add_subdirectory(acoustic_iso_float)

# include folder for pyAcousticIsotropic library
add_subdirectory(python/python_double)
add_subdirectory(python/python_float)

# include folder for Seismic utilities
add_subdirectory(seis_utils/seis_utils_double/)
add_subdirectory(seis_utils/seis_utils_float/)

# include folder for pySolver
add_subdirectory(external/pySolver/GenericSolver/)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${ACOUSTIC_ISOTROPIC_OPERATORS_PYBIND11_MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${ACOUSTIC_ISOTROPIC_OPERATORS_PYBIND11_VERSION}")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_NAME "Acoustic Isotropic Operators")
set(CPACK_PACKAGE_CONTACT "gbarnier@stanford.edu/ettore88@stanford.edu/sfarris@stanford.edu")
include(CPack)

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

#Required packages
find_package(Boost REQUIRED)
find_package(OpenMP)

#Adding genericIO submodule
# execute_process(COMMAND git submodule add http://cees-gitlab.Stanford.EDU/bob/genericIO.git external/genericIO WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
#Updating genericIO library
  execute_process(COMMAND git submodule update --init --recursive -- external/genericIO WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_subdirectory(external/genericIO)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/external/genericIO)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/external/genericIO/external/sepVector)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/cpp)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/sepVector/lib)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/sepVector/external/hypercube/lib)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/buffers/external/json)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/genericIO/external/buffers/lib)

#Finding multithread libraries
find_package(Threads)
# Adding TBB library
find_package(TBB REQUIRED)

#Acoustic isotropic operators (Double precision)
add_subdirectory(acoustic_iso_double)
# include folder for ACOUSTIC_ISO_DOUBLE library
include_directories(acoustic_iso_double/include)
# include folder for pyAcousticIsotropic library
# add_subdirectory(python)

include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${PROJECT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${ACOUSTIC_ISOTROPIC_OPERATORS_PYBIND11_MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${ACOUSTIC_ISOTROPIC_OPERATORS_PYBIND11_VERSION}")
set(CPACK_PACKAGE_VERSION_PATCH "0")
set(CPACK_PACKAGE_NAME "Acoustic Isotropic Operators")
set(CPACK_PACKAGE_CONTACT "gbarnier@stanford.edu/ettore88@stanford.edu/sfarris@stanford.edu")
include(CPack)

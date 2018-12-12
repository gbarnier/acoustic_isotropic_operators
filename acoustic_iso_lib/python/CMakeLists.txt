cmake_minimum_required(VERSION 2.8)
project(pyAcoustic_iso_double)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# This is a CMake example for Python
find_package(pybind11)
if (pybind11_FOUND)
  #PYBIND11 library headers
  include_directories(${PYTHON_INCLUDE_DIR})
  include_directories(${pybind11_INCLUDE_DIR})
  include_directories(${PYTHON_INCLUDE_PATH})
  #DERIVATIVE library headers
  include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../acoustic_isotropic_operators")

  find_package(PythonLibs)
  include_directories(${PYTHON_INCLUDE_PATH})

  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  include_directories(${GENERIC_EXPORT_INC_DIRS})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  set(JUNK ACOUSTIC_ISO_DOUBLE genericCpp buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

  #Compiling and adding module to library
  pybind11_add_module(pyAcoustic_iso_double MODULE pyAcoustic_iso_double.cpp)
  target_link_libraries(pyAcoustic_iso_double PRIVATE ${JUNK})

  #Installing library
  install( TARGETS pyAcoustic_iso_double DESTINATION lib/python)
  install( FILES Acoustic_iso_double.py DESTINATION lib/python)

else(pybind11_FOUND)
  MESSAGE("WARNING!Did not find pybid11. Not compiling ${PROJECT-NAME}")
endif(pybind11_FOUND)

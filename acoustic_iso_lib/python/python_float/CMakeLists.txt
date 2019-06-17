cmake_minimum_required(VERSION 2.8)
project(pyAcoustic_iso_float)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# include folder for ACOUSTIC_ISO_FLOAT library
include_directories(../../acoustic_iso_float/include)
include_directories(../../acoustic_iso_we_float/include)

# This is a CMake example for Python
find_package(pybind11)
if (pybind11_FOUND)
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
  pybind11_add_module(pyAcoustic_iso_float_nl MODULE pyAcoustic_iso_float_nl.cpp)
  target_link_libraries(pyAcoustic_iso_float_nl PRIVATE ACOUSTIC_ISO_FLOAT_NL ${LIBS})
  pybind11_add_module(pyAcoustic_iso_float_born MODULE pyAcoustic_iso_float_born.cpp)
  target_link_libraries(pyAcoustic_iso_float_born PRIVATE ACOUSTIC_ISO_FLOAT_BORN ${LIBS})
  pybind11_add_module(pyAcoustic_iso_float_born_ext MODULE pyAcoustic_iso_float_born_ext.cpp)
  target_link_libraries(pyAcoustic_iso_float_born_ext PRIVATE ACOUSTIC_ISO_FLOAT_BORN_EXT ${LIBS})
  pybind11_add_module(pyAcoustic_iso_float_tomo MODULE pyAcoustic_iso_float_tomo.cpp)
  target_link_libraries(pyAcoustic_iso_float_tomo PRIVATE ACOUSTIC_ISO_FLOAT_TOMO_EXT ${LIBS})
  pybind11_add_module(pyAcoustic_iso_float_wemva MODULE pyAcoustic_iso_float_wemva.cpp)
  target_link_libraries(pyAcoustic_iso_float_wemva PRIVATE ACOUSTIC_ISO_FLOAT_WEMVA_EXT ${LIBS})
  # pybind11_add_module(pyAcoustic_iso_float_we MODULE pyAcoustic_iso_float_we.cpp)
  # target_link_libraries(pyAcoustic_iso_float_we PRIVATE ACOUSTIC_ISO_FLOAT_WE ${LIBS})

  # Installing library
  install( TARGETS pyAcoustic_iso_float_nl DESTINATION lib/python) # Module for non-linear operator
  install( TARGETS pyAcoustic_iso_float_born DESTINATION lib/python) # Module for Born operator
  install( TARGETS pyAcoustic_iso_float_born_ext DESTINATION lib/python) # Module for Extended Born operator
  install( TARGETS pyAcoustic_iso_float_tomo DESTINATION lib/python) # Module for Extended Tomo operator
  install( TARGETS pyAcoustic_iso_float_wemva DESTINATION lib/python) # Module for Extended Wemva operator
  install( TARGETS pyAcoustic_iso_float_we DESTINATION lib/python) # Module for non-linear operator
  install( FILES Acoustic_iso_float.py DESTINATION lib/python)
  # install( FILES Acoustic_iso_float_we.py DESTINATION lib/python)
  install( FILES inversionUtils.py DESTINATION lib/python)


  # Installing Python executables
  install( FILES nonlinearPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES BornPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES BornExtPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES tomoExtPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES wemvaExtPythonFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES fwiMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES fwimeMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES lsrtmMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES extLsrtmMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES wemvaMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES tomoLinearMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES SymesPseudoInvMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES SymesExtLsrtmMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES fwimeFwiObjMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  # install( FILES waveEquationPythonAcousticFloatMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  # install( FILES waveEqInvFloat.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

else(pybind11_FOUND)
  MESSAGE("WARNING!Did not find pybid11. Not compiling ${PROJECT-NAME}")
endif(pybind11_FOUND)

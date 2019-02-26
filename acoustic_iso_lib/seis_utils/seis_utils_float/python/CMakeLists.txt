cmake_minimum_required(VERSION 2.8)
project(seis_utils_float)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(PythonMains
    waveletFloatMain.py
)

install(FILES ${PythonModules} DESTINATION lib/python)
install(FILES ${PythonMains} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)


# This is a CMake example for Python
find_package(pybind11)
if (pybind11_FOUND)

  # PYBIND11 library headers
  include_directories(${PYTHON_INCLUDE_DIR})
  include_directories(${pybind11_INCLUDE_DIR})
  include_directories(${PYTHON_INCLUDE_PATH})

  find_package(PythonLibs)
  include_directories(${PYTHON_INCLUDE_PATH})
  # Adding include files
  include_directories(../include)

  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  include_directories(${GENERIC_EXPORT_INC_DIRS})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  set(LIBS genericCpp sepVector buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

  # Compiling and adding module to library
  pybind11_add_module(pyDataTaper MODULE pyDataTaper.cpp)
  target_link_libraries(pyDataTaper PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
  pybind11_add_module(pyInterpBSpline2d MODULE pyInterpBSpline2d.cpp)
  target_link_libraries(pyInterpBSpline2d PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
  pybind11_add_module(pyDsoGpu MODULE pyDsoGpu.cpp)
  target_link_libraries(pyDsoGpu PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
  pybind11_add_module(pySpatialDeriv MODULE pySpatialDeriv.cpp)
  target_link_libraries(pySpatialDeriv PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})

  # Installing library
  install( TARGETS pyDataTaper DESTINATION lib/python)
  install( FILES dataTaperModule.py DESTINATION lib/python)
  install( TARGETS pyInterpBSpline2d DESTINATION lib/python)
  install( FILES interpBSpline2dModule.py DESTINATION lib/python)
  install( TARGETS pyDsoGpu DESTINATION lib/python)
  install( FILES dsoGpuModule.py DESTINATION lib/python)
  install( TARGETS pySpatialDeriv DESTINATION lib/python)
  install( FILES spatialDerivModule.py DESTINATION lib/python)

  # Installing Python executables
  install( FILES dataTaperMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES interpSpline2dMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES spatialDerivMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
  install( FILES dsoGpuMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
else(pybind11_FOUND)
  MESSAGE("WARNING!Did not find pybid11. Not compiling ${PROJECT-NAME}")
endif(pybind11_FOUND)

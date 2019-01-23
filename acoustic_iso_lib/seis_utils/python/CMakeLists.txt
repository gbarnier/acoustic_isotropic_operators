cmake_minimum_required(VERSION 2.8)
project(seis_utils)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(PythonMains
    waveletMain.py
)

set(PythonModules
    pyDso.py
)

install(FILES ${PythonModules} DESTINATION lib/python)
install(FILES ${PythonMains} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)


# This is a CMake example for Python
find_package(pybind11)
if (pybind11_FOUND)
  #PYBIND11 library headers
  include_directories(${PYTHON_INCLUDE_DIR})
  include_directories(${pybind11_INCLUDE_DIR})
  include_directories(${PYTHON_INCLUDE_PATH})

  find_package(PythonLibs)
  include_directories(${PYTHON_INCLUDE_PATH})
  #Adding include files
  include_directories(../include)

  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  include_directories(${GENERIC_EXPORT_INC_DIRS})
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

  set(LIBS genericCpp buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

  #Compiling and adding module to library
  pybind11_add_module(pydataTaper MODULE pydataTaper.cpp)
  target_link_libraries(pydataTaper PRIVATE SEIS_UTILS_LIB ${LIBS})

  #Installing library
  install( TARGETS pydataTaper DESTINATION lib/python) #Module for non-linear operator
  #Installing Python executables
  install( FILES dataTaperMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

else(pybind11_FOUND)
  MESSAGE("WARNING!Did not find pybid11. Not compiling ${PROJECT-NAME}")
endif(pybind11_FOUND)

cmake_minimum_required(VERSION 2.8)
project(seis_utils_double)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(PythonMains

)

set(PythonModules
    pyDsoDouble.py
)

set(PYTHON_DIR "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
install(FILES ${PythonModules} DESTINATION ${PYTHON_DIR})
# install(FILES ${PythonMains} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

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

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
                               ${hypercube_BINARY_DIR}
                               ${buffers-lib_BINARY_DIR}
                               ${json_BINARY_DIR}
                               ${sepVector-lib_BINARY_DIR}
                               ${SEPlib_LIBDIR}
)

set(LIBS genericCpp sepVector buffers jsonCpp hypercube  ${OpenMP_EXE_LINKER_FLAGS} ${TBB_LIBRARIES} ${GENERIC_EXPORT_LIBS})

# Compiling and adding module to library
pybind11_add_module(pydataTaperDouble MODULE pydataTaperDouble.cpp)
target_link_libraries(pydataTaperDouble PRIVATE SEIS_UTILS_DOUBLE_LIB ${LIBS})
pybind11_add_module(pyInterpBSpline1dDouble MODULE pyInterpBSpline1dDouble.cpp)
target_link_libraries(pyInterpBSpline1dDouble PRIVATE SEIS_UTILS_DOUBLE_LIB ${LIBS})
pybind11_add_module(pyInterpBSpline2dDouble MODULE pyInterpBSpline2dDouble.cpp)
target_link_libraries(pyInterpBSpline2dDouble PRIVATE SEIS_UTILS_DOUBLE_LIB ${LIBS})

# Installing library
install( TARGETS pydataTaperDouble DESTINATION ${PYTHON_DIR})
install( TARGETS pyInterpBSpline1dDouble DESTINATION ${PYTHON_DIR})
install( TARGETS pyInterpBSpline2dDouble DESTINATION ${PYTHON_DIR})
install( FILES dataTaperDoubleModule.py DESTINATION ${PYTHON_DIR})
install( FILES interpBSpline1dDoubleModule.py DESTINATION ${PYTHON_DIR})
install( FILES interpBSpline2dDoubleModule.py DESTINATION ${PYTHON_DIR})

#Installing Python executables
install( FILES dataTaperDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES interpBSpline1dDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES interpBSpline2dDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

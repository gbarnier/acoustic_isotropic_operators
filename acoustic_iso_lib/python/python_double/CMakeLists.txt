cmake_minimum_required(VERSION 2.8)
project(pyAcoustic_iso_double)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

# include folder for ACOUSTIC_ISO_DOUBLE library
include_directories(../../acoustic_iso_double/include)

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

#Compiling and adding module to library
pybind11_add_module(pyAcoustic_iso_double_nl MODULE pyAcoustic_iso_double_nl.cpp)
target_link_libraries(pyAcoustic_iso_double_nl PRIVATE ACOUSTIC_ISO_DOUBLE_NL ${LIBS})
pybind11_add_module(pyAcoustic_iso_double_born MODULE pyAcoustic_iso_double_born.cpp)
target_link_libraries(pyAcoustic_iso_double_born PRIVATE ACOUSTIC_ISO_DOUBLE_BORN ${LIBS})
pybind11_add_module(pyAcoustic_iso_double_born_ext MODULE pyAcoustic_iso_double_born_ext.cpp)
target_link_libraries(pyAcoustic_iso_double_born_ext PRIVATE ACOUSTIC_ISO_DOUBLE_BORN_EXT ${LIBS})
pybind11_add_module(pyAcoustic_iso_double_tomo MODULE pyAcoustic_iso_double_tomo.cpp)
target_link_libraries(pyAcoustic_iso_double_tomo PRIVATE ACOUSTIC_ISO_DOUBLE_TOMO_EXT ${LIBS})
pybind11_add_module(pyAcoustic_iso_double_wemva MODULE pyAcoustic_iso_double_wemva.cpp)
target_link_libraries(pyAcoustic_iso_double_wemva PRIVATE ACOUSTIC_ISO_DOUBLE_WEMVA_EXT ${LIBS})

#Installing library
install( TARGETS pyAcoustic_iso_double_nl DESTINATION lib/python) #Module for non-linear operator
install( TARGETS pyAcoustic_iso_double_born DESTINATION lib/python) #Module for Born operator
install( TARGETS pyAcoustic_iso_double_born_ext DESTINATION lib/python) #Module for Extended Born operator
install( TARGETS pyAcoustic_iso_double_tomo DESTINATION lib/python) #Module for Extended Tomo operator
install( TARGETS pyAcoustic_iso_double_wemva DESTINATION lib/python) #Module for Extended Wemva operator
install( FILES Acoustic_iso_double.py DESTINATION lib/python)
#Installing Python executables
install( FILES nonlinearPythonDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES BornPythonDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES BornExtPythonDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES tomoExtPythonDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES wemvaExtPythonDoubleMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

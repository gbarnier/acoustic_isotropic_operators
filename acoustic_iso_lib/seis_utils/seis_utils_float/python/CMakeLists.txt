cmake_minimum_required(VERSION 2.8)
project(seis_utils_float)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)

set(PythonMains
    waveletFloatMain.py
    fftFwiMain.py
)

install(FILES ${PythonModules} DESTINATION lib/python)
install(FILES ${PythonMains} PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

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
pybind11_add_module(pyDataTaper MODULE pyDataTaper.cpp)
target_link_libraries(pyDataTaper PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
pybind11_add_module(pyInterpBSpline1d MODULE pyInterpBSpline1d.cpp)
target_link_libraries(pyInterpBSpline1d PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
pybind11_add_module(pyInterpBSpline2d MODULE pyInterpBSpline2d.cpp)
target_link_libraries(pyInterpBSpline2d PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
pybind11_add_module(pyInterpBSpline3d MODULE pyInterpBSpline3d.cpp)
target_link_libraries(pyInterpBSpline3d PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
pybind11_add_module(pyDsoGpu MODULE pyDsoGpu.cpp)
target_link_libraries(pyDsoGpu PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
pybind11_add_module(pyDsoInvGpu MODULE pyDsoInvGpu.cpp)
target_link_libraries(pyDsoInvGpu PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
pybind11_add_module(pySpatialDeriv MODULE pySpatialDeriv.cpp)
target_link_libraries(pySpatialDeriv PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
pybind11_add_module(pyInterpRbf1d MODULE pyInterpRbf1d.cpp)
target_link_libraries(pyInterpRbf1d PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})
pybind11_add_module(pyTimeInteg MODULE pyTimeInteg.cpp)
target_link_libraries(pyTimeInteg PRIVATE SEIS_UTILS_FLOAT_LIB ${LIBS})

# Installing library
install( TARGETS pyDataTaper DESTINATION lib/python)
install( FILES dataTaperModule.py DESTINATION lib/python)
install( TARGETS pyInterpBSpline1d DESTINATION lib/python)
install( TARGETS pyInterpBSpline2d DESTINATION lib/python)
install( TARGETS pyInterpBSpline3d DESTINATION lib/python)
install( FILES interpBSplineModule.py DESTINATION lib/python)
install( TARGETS pyDsoGpu DESTINATION lib/python)
install( FILES dsoGpuModule.py DESTINATION lib/python)
install( TARGETS pyDsoInvGpu DESTINATION lib/python)
install( FILES dsoInvGpuModule.py DESTINATION lib/python)
install( TARGETS pySpatialDeriv DESTINATION lib/python)
install( FILES spatialDerivModule.py DESTINATION lib/python)
install( TARGETS pyInterpRbf1d DESTINATION lib/python)
install( FILES interpRbf1dModule.py DESTINATION lib/python)
install( FILES ShotRecTaperModule.py DESTINATION lib/python)
install( FILES maskGradientModule.py DESTINATION lib/python)
install( TARGETS pyTimeInteg DESTINATION lib/python)
install( FILES timeIntegModule.py DESTINATION lib/python)
install( FILES phaseOnlyXkModule.py DESTINATION lib/python)
install( FILES off2angModule.py DESTINATION lib/python)

# Installing Python executables
install( FILES dataTaperMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES derivTimeMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES maskGradientMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES interpBSplineMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES interpBSplineIterMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES spatialDerivMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES dsoGpuMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES dsoInvGpuMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES interpBSplineInvMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES dataCompMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES timeIntegMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES fftFwiMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES phaseOnlyXkMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES phaseOnlyXkJacMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)
install( FILES off2angMain.py PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION bin)

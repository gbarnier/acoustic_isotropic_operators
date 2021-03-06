cmake_minimum_required(VERSION 2.8)
project(ACOUSTIC_ISO_DOUBLE-lib)

#Adding include files
include_directories(include)
include_directories(${Boost_INCLUDE_DIR})
# include_directories(${CMAKE_BUILD_DIR}/../external/ioLibs/external/genericIO/external/sepVector/lib/)

#Header files to include during installation
set(INC_DIR include)
set(HEADERS ${INC_DIR}/deviceGpu.h
			      ${INC_DIR}/fdParam.h
			      ${INC_DIR}/nonlinearPropGpu.h
			      ${INC_DIR}/interpTimeLinTbb.h
			      ${INC_DIR}/nonlinearPropShotsGpu.h
			      ${INC_DIR}/secondTimeDerivative.h
						${INC_DIR}/BornExtGpu.h
						${INC_DIR}/BornExtShotsGpu.h
						${INC_DIR}/BornExtShotsGpuFunctions.h
		)

#Files to be compiled
set(SRC_DIR src)
#Files for nonLinear operator library
set(SOURCE_NL ${SRC_DIR}/nonlinearPropShotsGpu.cpp
		   	  ${SRC_DIR}/nonlinearPropGpu.cpp
	   	   	  ${SRC_DIR}/fdParam.cpp
	   	      ${SRC_DIR}/deviceGpu.cpp
	   	      ${SRC_DIR}/nonlinearShotsGpuFunctions.cu
	   	      ${SRC_DIR}/interpTimeLinTbb.cpp
		      ${SRC_DIR}/secondTimeDerivative.cpp
)
#Files for Born operator library
set(SOURCE_BORN ${SRC_DIR}/fdParam.cpp
	   	    	${SRC_DIR}/deviceGpu.cpp
	   	        ${SRC_DIR}/interpTimeLinTbb.cpp
		    	${SRC_DIR}/secondTimeDerivative.cpp
		    	${SRC_DIR}/BornShotsGpuFunctions.cu
				${SRC_DIR}/BornShotsGpu.cpp
		    	${SRC_DIR}/BornGpu.cpp
)
#Files for Extended Born operator library
set(SOURCE_BORN_EXT ${SRC_DIR}/fdParam.cpp
	   	        	${SRC_DIR}/deviceGpu.cpp
	   	        	${SRC_DIR}/interpTimeLinTbb.cpp
		        	${SRC_DIR}/secondTimeDerivative.cpp
		        	${SRC_DIR}/BornExtShotsGpuFunctions.cu
		        	${SRC_DIR}/BornExtShotsGpu.cpp
		        	${SRC_DIR}/BornExtGpu.cpp
)

#Files for Extended tomo operator library
set(SOURCE_TOMO_EXT ${SRC_DIR}/fdParam.cpp
	   	        	${SRC_DIR}/deviceGpu.cpp
	   	        	${SRC_DIR}/interpTimeLinTbb.cpp
		        	${SRC_DIR}/secondTimeDerivative.cpp
		        	${SRC_DIR}/tomoExtShotsGpuFunctions.cu
		        	${SRC_DIR}/tomoExtShotsGpu.cpp
		        	${SRC_DIR}/tomoExtGpu.cpp
)

#Files for Extended wemva operator library
set(SOURCE_WEMVA_EXT ${SRC_DIR}/fdParam.cpp
	   	        	${SRC_DIR}/deviceGpu.cpp
	   	        	${SRC_DIR}/interpTimeLinTbb.cpp
		        	${SRC_DIR}/secondTimeDerivative.cpp
		        	${SRC_DIR}/wemvaExtShotsGpuFunctions.cu
		        	${SRC_DIR}/wemvaExtShotsGpu.cpp
		        	${SRC_DIR}/wemvaExtGpu.cpp
)

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
				 ${hypercube_BINARY_DIR}
				 ${buffers-lib_BINARY_DIR}
				 ${json_BINARY_DIR}
				 ${sepVector-lib_BINARY_DIR}
				 ${SEPlib_LIBDIR}
)

#Setting io libraries
set(LIBS genericCpp sepVector sep3d sep)

#Compiling the project library
add_library(ACOUSTIC_ISO_DOUBLE_NL ${SOURCE_NL})
target_link_libraries(ACOUSTIC_ISO_DOUBLE_NL ${LIBS} OpenMP::OpenMP_CXX)
add_library(ACOUSTIC_ISO_DOUBLE_BORN ${SOURCE_BORN})
target_link_libraries(ACOUSTIC_ISO_DOUBLE_BORN ${LIBS} OpenMP::OpenMP_CXX)
add_library(ACOUSTIC_ISO_DOUBLE_BORN_EXT ${SOURCE_BORN_EXT})
target_link_libraries(ACOUSTIC_ISO_DOUBLE_BORN_EXT ${LIBS} OpenMP::OpenMP_CXX)
add_library(ACOUSTIC_ISO_DOUBLE_TOMO_EXT ${SOURCE_TOMO_EXT})
target_link_libraries(ACOUSTIC_ISO_DOUBLE_TOMO_EXT ${LIBS} OpenMP::OpenMP_CXX)
add_library(ACOUSTIC_ISO_DOUBLE_WEMVA_EXT ${SOURCE_WEMVA_EXT})
target_link_libraries(ACOUSTIC_ISO_DOUBLE_WEMVA_EXT ${LIBS} OpenMP::OpenMP_CXX)

# Compiling the project executables
# Non-linear operator
# add_executable(nonlinearPropShotsDoubleGpuMain ${SRC_DIR}/nonlinearPropShotsDoubleGpuMain.cpp)
# target_link_libraries(nonlinearPropShotsDoubleGpuMain ACOUSTIC_ISO_DOUBLE_NL)

# Born operator
# add_executable(BornShotsDoubleGpuMain ${SRC_DIR}/BornShotsDoubleGpuMain.cpp)
# target_link_libraries(BornShotsDoubleGpuMain ACOUSTIC_ISO_DOUBLE_BORN)

# Extended Born operator
# add_executable(BornExtShotsDoubleGpuMain ${SRC_DIR}/BornExtShotsDoubleGpuMain.cpp)
# target_link_libraries(BornExtShotsDoubleGpuMain ACOUSTIC_ISO_DOUBLE_BORN_EXT)

#Extended Tomo operator
# add_executable(tomoExtShotsDoubleGpuMain ${SRC_DIR}/tomoExtShotsDoubleGpuMain.cpp)
# target_link_libraries(tomoExtShotsDoubleGpuMain ACOUSTIC_ISO_DOUBLE_TOMO_EXT)
#Extended Tomo operator
# add_executable(wemvaExtShotsDoubleGpuMain ${SRC_DIR}/wemvaExtShotsDoubleGpuMain.cpp)
# target_link_libraries(wemvaExtShotsDoubleGpuMain ACOUSTIC_ISO_DOUBLE_WEMVA_EXT)
# Model padding program for GPU
# add_executable(padFileDoubleGpuMain ${SRC_DIR}/padFileDoubleGpuMain.cpp)
# target_link_libraries(padFileDoubleGpuMain ${LIBS} OpenMP::OpenMP_CXX)

#Adding libray and header files to install destination
install(TARGETS ACOUSTIC_ISO_DOUBLE_NL DESTINATION lib)
# install(TARGETS nonlinearPropShotsDoubleGpuMain DESTINATION bin)
install(TARGETS ACOUSTIC_ISO_DOUBLE_BORN DESTINATION lib)
# install(TARGETS BornShotsDoubleGpuMain DESTINATION bin)
install(TARGETS ACOUSTIC_ISO_DOUBLE_BORN_EXT DESTINATION lib)
# install(TARGETS BornExtShotsDoubleGpuMain DESTINATION bin)
install(TARGETS ACOUSTIC_ISO_DOUBLE_TOMO_EXT DESTINATION lib)
# install(TARGETS tomoExtShotsDoubleGpuMain DESTINATION bin)
install(TARGETS ACOUSTIC_ISO_DOUBLE_WEMVA_EXT DESTINATION lib)
# install(TARGETS wemvaExtShotsDoubleGpuMain DESTINATION bin)
# install(TARGETS padFileDoubleGpuMain DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)

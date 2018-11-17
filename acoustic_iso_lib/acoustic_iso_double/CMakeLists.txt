cmake_minimum_required(VERSION 2.8)
project(ACOUSTIC_ISO_DOUBLE-lib)

#Adding include files
include_directories(${hypercube_DIRNAME_include})
include_directories(${genericIO_DIRNAME_include})
include_directories(include)

#Files to be compiled
set(SRC_DIR src)
set(SOURCE ${SRC_DIR}/nonlinearPropShotsGpu.cpp
		   ${SRC_DIR}/nonlinearPropGpu.cpp
	   	   ${SRC_DIR}/fdParam.cpp
	   	   ${SRC_DIR}/deviceGpu.cpp
	   	   ${SRC_DIR}/nonlinearShotsGpuFunctions.cu
	   	   ${SRC_DIR}/interpTimeLinTbb.cpp
		   ${SRC_DIR}/secondTimeDerivative.cpp
	   )
#Executables to be compiled
set(EXECS  ${SRC_DIR}/nonlinearPropShotsGpuMain.cpp)

#Header files to include during installation
set(INC_DIR include)
set(HEADERS ${INC_DIR}/deviceGpu.h
			${INC_DIR}/fdParam.h
			${INC_DIR}/nonlinearPropGpu.h
			${INC_DIR}/interpTimeLinTbb.h
			${INC_DIR}/nonlinearPropShotsGpu.h
			${INC_DIR}/secondTimeDerivative.h
		)

#Compiling the project library
add_library(ACOUSTIC_ISO_DOUBLE ${SOURCE})
#Compiling the project executables
add_executable(nonlinearPropShotsGpuMain ${EXECS})
#Adding libray and header files to install destination
install(TARGETS ACOUSTIC_ISO_DOUBLE DESTINATION lib)
install(TARGETS nonlinearPropShotsGpuMain DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)

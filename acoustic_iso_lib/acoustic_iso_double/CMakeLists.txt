cmake_minimum_required(VERSION 2.8)
project(ACOUSTIC_ISO_DOUBLE-lib)

#Adding include files
include_directories(include)

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
#Executables to be compiled
# set(EXECS ${SRC_DIR}/nonlinearPropShotsGpuMain.cpp ${SRC_DIR}/BornShotsGpuMain.cpp ${SRC_DIR}/BornExtShotsGpuMain.cpp)

#Header files to include during installation
# set(INC_DIR include)
# set(HEADERS ${INC_DIR}/deviceGpu.h
# 			${INC_DIR}/fdParam.h
# 			${INC_DIR}/seismicOperator2D.h
# 			${INC_DIR}/nonlinearPropGpu.h
# 			${INC_DIR}/interpTimeLinTbb.h
# 			${INC_DIR}/nonlinearPropShotsGpu.h
# 			${INC_DIR}/secondTimeDerivative.h
# 			${INC_DIR}/BornShotsGpu.h
# 			${INC_DIR}/BornShotsGpuFunctions.h
# 			${INC_DIR}/BornGpu.h
# 			${INC_DIR}/BornExtShotsGpu.h
# 			${INC_DIR}/BornExtShotsGpuFunctions.h
# 			${INC_DIR}/BornExtGpu.h
# )

#SEP LIBRARY DIRECTORIES
link_directories(${Generic-cpp_BINARY_DIR}
				 ${hypercube_BINARY_DIR}
				 ${buffers-lib_BINARY_DIR}
				 ${json_BINARY_DIR}
				 ${sepVector-lib_BINARY_DIR}
				 /net/server/homes/sep/ettore/research/packages/acoustic_isotropic_operators/iolibs/local/lib/
)

#Compiling the project library
add_library(ACOUSTIC_ISO_DOUBLE_NL ${SOURCE_NL})
add_library(ACOUSTIC_ISO_DOUBLE_BORN ${SOURCE_BORN})
add_library(ACOUSTIC_ISO_DOUBLE_BORN_EXT ${SOURCE_BORN_EXT})
# Compiling the project executables
#Non-linear operator
add_executable(nonlinearPropShotsGpuMain ${SRC_DIR}/nonlinearPropShotsGpuMain.cpp)
target_link_libraries(nonlinearPropShotsGpuMain ACOUSTIC_ISO_DOUBLE_NL genericCpp OpenMP::OpenMP_CXX)
#Born operator
add_executable(BornShotsGpuMain ${SRC_DIR}/BornShotsGpuMain.cpp)
target_link_libraries(BornShotsGpuMain ACOUSTIC_ISO_DOUBLE_BORN genericCpp OpenMP::OpenMP_CXX)
#Exetended Born operator
add_executable(BornExtShotsGpuMain ${SRC_DIR}/BornExtShotsGpuMain.cpp)
target_link_libraries(BornExtShotsGpuMain ACOUSTIC_ISO_DOUBLE_BORN_EXT genericCpp OpenMP::OpenMP_CXX)
#Adding libray and header files to install destination
install(TARGETS ACOUSTIC_ISO_DOUBLE_NL DESTINATION lib)
install(TARGETS nonlinearPropShotsGpuMain DESTINATION bin)
install(TARGETS ACOUSTIC_ISO_DOUBLE_BORN DESTINATION lib)
install(TARGETS BornShotsGpuMain DESTINATION bin)
install(TARGETS ACOUSTIC_ISO_DOUBLE_BORN_EXT DESTINATION lib)
install(TARGETS BornExtShotsGpuMain DESTINATION bin)
install(FILES ${HEADERS} DESTINATION include)

##DESCRIPTION
Note use cmake 3.14

##COMPILATION
When the package is cloned, run the following command once:
```
git submodule update --init --recursive --remote elastic_iso_double_lib/external/genericIO

```

Before compiling the library, SEPIO must be installed. To do so, change the pathtoSEPIO and run the following commands:
```
git clone http://cees-gitlab.Stanford.EDU/SEP-external/sep-iolibs.git pathtoSEPIO/iolibs/src

mkdir -p pathtoSEPIO/iolibs/build

cd pathtoSEPIO/iolibs/build

cmake -DCMAKE_INSTALL_PREFIX=../local ../src

make install
```
If SEPIO is already installed, then skip the previous step.

Change pathtoSEPIO and run to build the library:
```
cd build

cmake -DCMAKE_INSTALL_PREFIX=folder_for_buiding -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc -DSEPlib_LIBDIR=pathtoSEPIO/iolibs/local/lib/ -DSEPlib_DIR=pathtoSEPIO/iolibs/local/cmake/SEP -DCMAKE_MODULE_PREFIX=pathtoSEPIO/iolibs/local/cmake -DCMAKE_BUILD_TYPE=Debug ../acoustic_iso_lib/

make install
```

If the previous cmake command does not work use
```
cmake -DCMAKE_INSTALL_PREFIX=folder_for_buiding -DSEPlib_LIBDIR=pathtoSEPIO/iolibs/local/lib/ -DSEPlib_DIR=pathtoSEPIO/iolibs/local/cmake/SEP -DCMAKE_MODULE_PREFIX=pathtoSEPIO/iolibs/local/cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc -Dpybind11_DIR=${PWD}/../acoustic_iso_lib/cmake/ -DPYTHON_EXECUTABLE=/usr/local/bin/python3.5 -DCMAKE_BUILD_TYPE=Debug ../acoustic_iso_lib
```

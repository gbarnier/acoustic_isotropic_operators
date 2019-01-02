##DESCRIPTION

##COMPILATION
Before compiling the library, SEPIO must be installed. To do so, change the pathtoSEPIO and run the following commands:
```
git clone http://cees-gitlab.Stanford.EDU/SEP-external/sep-iolibs.git pathtoSEPIO/iolibs/src

mdkir -p pathtoSEPIO/iolibs/build

cd pathtoSEPIO/iolibs/build

cmake -DCMAKE_INSTALL_PREFIX=pathtoSEPIO/iolibs/local ../src

make install
```
If SEPIO is already installed, then skip the previous step.

Change pathtoSEPIO and run to build the library:
```
cd build

cmake -DCMAKE_INSTALL_PREFIX=folder_for_buiding -DSEPlib_DIR=pathtoSEPIO/iolibs/local/cmake -DCMAKE_MODULE_PREFIX=pathtoSEPIO/iolibs/local/cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-9.0/bin/nvcc ../acoustic_iso_lib

make install
```

##DESCRIPTION 

##COMPILATION
To build library run:
```
cd build

cmake -DCMAKE_INSTALL_PREFIX=folder_for_buiding -DSEPlib_DIR=/opt/SEP/SEP8_jarvis/cmake -DCMAKE_MODULE_PREFIX=/opt/SEP/SEP8_jarvis/cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda-9.0/bin/nvcc ../acoustic_iso_lib

make install
```

##DESCRIPTION
Note use cmake 3.14 or higher

##COMPILATION

When the package is cloned, run the following command once:
```
git submodule update --init --recursive -- acoustic_iso_lib/external/ioLibs
git submodule update --init --recursive -- acoustic_iso_lib/external/pySolver

```

To build library run:
```
cd build

cmake -DCMAKE_INSTALL_PREFIX=installation_path -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc -DBUILD_SHARED_LIBS=True ../acoustic_iso_lib/

make install -j8

```

##INSTALLATION USING CONDA

```
# Install conda
# Follow the easy instructions at https://docs.conda.io/projects/conda/en/latest/user-guide/install/
# or simply use the following two commands
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

# Creating necessary environment
conda create -n EGS
conda activate EGS
conda install -c anaconda cmake
conda install -c conda-forge flex
conda install -c anaconda boost
conda install -c statiskit libboost-dev
conda install -c conda-forge tbb tbb-devel
conda install -c conda-forge pybind11
conda install dask
conda install -c conda-forge dask-jobqueue
conda install -c anaconda scipy
conda install -c conda-forge matplotlib
conda install -c anaconda jupyter
conda install h5py
conda install -c conda-forge scikit-build
conda install -c conda-forge setuptools_scm
conda install -c anaconda pytest
conda install numba
# The code installation requires that C++, CC, FORTRAN, and CUDA compilers are installed (e.g., g++, gcc, gfortran, nvcc)

# Installing GPU-wave-equation library
git clone http://zapad.Stanford.EDU/barnier/acoustic_isotropic_operators.git
cd acoustic_isotropic_operators
git submodule update --init --recursive -- acoustic_iso_lib/external/ioLibs
git submodule update --init --recursive -- acoustic_iso_lib/external/pySolver
git submodule update --init --recursive -- acoustic_iso_lib/external/pybind11
mkdir build
cd build

# Now try to run the following cmake command
cmake -DCMAKE_INSTALL_PREFIX=../local ../acoustic_iso_lib/ -DBUILD_SHARED_LIBS=True
# If this command breaks, try to set compiler paths manually
cmake -DCMAKE_INSTALL_PREFIX=../local -DCMAKE_CUDA_COMPILER=PATH-TO-NVCC -DCMAKE_CXX_COMPILER=PATH-TO-C++-COMPILER -DCMAKE_C_COMPILER=PATH-TO-CC-COMPILER -DCMAKE_Fortran_COMPILER=PATH-TO-FORTRAN-COMPILER -DPYTHON_EXECUTABLE=${CONDA_PREFIX}/bin/python3 -DBUILD_SHARED_LIBS=True ../acoustic_iso_lib/
# If that also breaks, run the following conda command, manually set the nvcc full path, and run
conda install gcc_linux-64 gxx_linux-64 gfortran_linux-64
cmake -DCMAKE_INSTALL_PREFIX=../local -DCMAKE_CUDA_COMPILER=PATH-TO-NVCC ../acoustic_iso_lib/ -DCMAKE_CXX_COMPILER=`ls ${CONDA_PREFIX}/bin/*g++` -DCMAKE_C_COMPILER=`ls ${CONDA_PREFIX}/bin/*gcc` -DCMAKE_Fortran_COMPILER=`ls ${CONDA_PREFIX}/bin/*gfortran` -DPYTHON_EXECUTABLE=${CONDA_PREFIX}/bin/python3 -DBUILD_SHARED_LIBS=True

# Now let's install the library. If it breaks, try to set different compilers using the previous commands
make install -j16
cd ..

# Setting module file
sed -i  's|path-to-EGSlib|'$PWD'|g' module/EGSlib
sed -i  's|MAJOR.MINOR|'`python3 -V | colrm 1 7 | colrm 4`'|g' module/EGSlib

###################################################################
# Now edit the file EGSlib in the folder module                   #
# The user needs to create a folder for their binary files        #
# and change the path-to-folder-to-binary-files/scratch on line 26#
###################################################################

# Changing activation and deactivation env_vars
touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo '#!/bin/sh' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "module use ${PWD}/module" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo "module load EGSlib"  >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

conda deactivate
conda activate EGS

touch $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo '#!/bin/sh' >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo "module unload EGSlib"  >> $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh

conda activate EGS
```

##Uninstallation of the library
```
rm -rf acoustic_isotropic_operators
# Remove EGS env
conda remove --name EGS --all
```

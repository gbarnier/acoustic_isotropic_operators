##DESCRIPTION
Note use cmake 3.14

##COMPILATION

When the package is cloned, run the following command once:
```
git submodule update --init --recursive -- acoustic_iso_lib/external/ioLibs
git submodule update --init --recursive -- acoustic_iso_lib/external/pySolver

```

To build library run:
```
cd build

cmake -DCMAKE_INSTALL_PREFIX=installation_path -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc ../acoustic_iso_lib/

make install

```

##INSTALLATION USING CONDA

```
# Install conda
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

# Creating necessary environment
git clone http://zapad.Stanford.EDU/barnier/acoustic_isotropic_operators.git
cd acoustic_isotropic_operators
conda env create -f environment.yml
# If the previous command fails with your conda install try the following command
# conda create --name EGS --file spec-file.txt
conda activate EGS

# Installing GPU-wave-equation library
git submodule update --init --recursive -- acoustic_iso_lib/external/ioLibs
git submodule update --init --recursive -- acoustic_iso_lib/external/pySolver
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../local -DCMAKE_CUDA_COMPILER=${CONDA_PREFIX}/bin/nvcc ../acoustic_iso_lib/ -DCMAKE_CXX_COMPILER=${CONDA_PREFIX}/bin/g++ -DCMAKE_C_COMPILER=${CONDA_PREFIX}/bin/gcc -DCMAKE_Fortran_COMPILER=${CONDA_PREFIX}/bin/gfortran -DPYTHON_EXECUTABLE=${CONDA_PREFIX}/bin/python3
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

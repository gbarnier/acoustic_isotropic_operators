#include <cstring>
#include <iostream>
#include "waveEquationAcousticGpuFunctions.h"
#include "varDeclareWaveEquation.h"
#include "kernelsGpuWaveEquationAcoustic.cu"
#include "cudaErrors.cu"
#include <cstring>
#include <assert.h>

void initWaveEquationAcousticGpu(float dz, float dx, int nz, int nx, int nts, float dts, int minPad, int blockSize, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU
  cudaSetDevice(iGpuId);

  // Host variables
	host_nz = nz;
	host_nx = nx;
	host_dz = dz;
	host_dx = dx;
	host_nts = nts;
	host_dts = dts;

	// /**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {
		// Array of pointers to wavefields
		dev_p0 = new float*[nGpu];
		dev_p1 = new float*[nGpu];

		// Scaled earth parameters velocity
		dev_slsqDt2= new float*[nGpu];
		// cosine dampening 
		dev_cosDamp = new float*[nGpu];
  }

  /**************************** COMPUTE DERIVATIVE COEFFICIENTS ************************/
  float zCoeff[COEFF_SIZE];
	float xCoeff[COEFF_SIZE];

  	zCoeff[0] = -2.927222222 / (dz * dz);
  	zCoeff[1]=  1.666666667 / (dz * dz);
  	zCoeff[2]= -0.238095238 / (dz * dz);
  	zCoeff[3]=  0.039682539 / (dz * dz);
  	zCoeff[4]= -0.004960317 / (dz * dz);
  	zCoeff[5]=  0.000317460 / (dz * dz);

  	xCoeff[0]= -2.927222222 / (dx * dx);
  	xCoeff[1]=  1.666666667 / (dx * dx);
  	xCoeff[2]= -0.238095238 / (dx * dx);
  	xCoeff[3]=  0.039682539 / (dx * dx);
  	xCoeff[4]= -0.004960317 / (dx * dx);
  	xCoeff[5]=  0.000317460 / (dx * dx);

  /************************** COMPUTE COSINE DAMPING COEFFICIENTS **********************/

	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR: Padding value is too high ****" << std::endl;
		assert (1==2);
	}
	float cosDampingCoeff[minPad];
	float alphaCos=0.99;
	// Cosine padding
	for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
		float arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		cosDampingCoeff[iFilter-FAT] = arg;
	}
  /************************* COPY TO CONSTANT MEMORY **********************/
  // Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_zCoeff, zCoeff, COEFF_SIZE*sizeof(float), 0, cudaMemcpyHostToDevice)); // Copy derivative coefficients to device
	cuda_call(cudaMemcpyToSymbol(dev_xCoeff, xCoeff, COEFF_SIZE*sizeof(float), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_minPad, &minPad, sizeof(int), 0, cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nz, &nz, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	// cuda_call(cudaMemcpyToSymbol(dev_dx, &host_dx, sizeof(float), 0, cudaMemcpyHostToDevice));
	// cuda_call(cudaMemcpyToSymbol(dev_dz, &host_dz, sizeof(float), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_dts, &host_dts, sizeof(float), 0, cudaMemcpyHostToDevice));
	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(float), 0, cudaMemcpyHostToDevice)); // Array for damping
}

void allocateWaveEquationAcousticGpu(float *slsqDt2, float *cosDamp, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample){

	// Set GPU
  cudaSetDevice(iGpuId);

	// Allocate scaled elastic parameters to device
	cuda_call(cudaMalloc((void**) &dev_slsqDt2[iGpu], host_nz*host_nx*sizeof(float)));
	// Allocate scaled elastic parameters to device
	cuda_call(cudaMalloc((void**) &dev_cosDamp[iGpu], host_nz*host_nx*sizeof(float)));

  // Copy scaled elastic parameters to device
	cuda_call(cudaMemcpy(dev_slsqDt2[iGpu], slsqDt2, host_nz*host_nx*sizeof(float), cudaMemcpyHostToDevice));
  // Copy cosing dampening matrix to device
	cuda_call(cudaMemcpy(dev_cosDamp[iGpu], cosDamp, host_nz*host_nx*sizeof(float), cudaMemcpyHostToDevice));

	// Allocate wavefields on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*sizeof(float)));
  cuda_call(cudaMalloc((void**) &dev_p1[iGpu], (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*sizeof(float)));

}

void waveEquationAcousticFwdGpu(float *model,float *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample){

	// Set GPU
  cudaSetDevice(iGpuId);
	//copy model to gpu
	cuda_call(cudaMemcpy(dev_p1[iGpu], model+firstTimeSample*host_nz*host_nx, (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*sizeof(float), cudaMemcpyHostToDevice));

	//copy data to gpu
	cuda_call(cudaMemcpy(dev_p0[iGpu], data+firstTimeSample*host_nz*host_nx, (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*sizeof(float), cudaMemcpyHostToDevice));


	//call fwd gpu kernel
	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	int nblockz = ((lastTimeSample-firstTimeSample+1)+BLOCK_SIZE) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky, nblockz);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

	kernel_exec(ker_we_fwd<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_slsqDt2[iGpu],dev_cosDamp[iGpu], firstTimeSample,lastTimeSample));
	kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu],firstTimeSample,lastTimeSample));

	//first, last, and other gpus have different copies
	if(firstTimeSample==0 && lastTimeSample==host_nts-1) {
		//include first and last sample in block
		cuda_call(cudaMemcpy(data, dev_p0[iGpu], host_nts*host_nz*host_nx*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else if(firstTimeSample==0){
		//exclude last sample | include first sample in block
		cuda_call(cudaMemcpy(data, dev_p0[iGpu], (lastTimeSample-firstTimeSample)*host_nz*host_nx*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else if(lastTimeSample==host_nts-1){
		//exclude first sample | include last sample in block
		cuda_call(cudaMemcpy(data+(firstTimeSample+1)*host_nz*host_nx, dev_p0[iGpu]+1*host_nz*host_nx, (lastTimeSample-firstTimeSample)*host_nz*host_nx*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else{
		//exclude first and last sample in block
		cuda_call(cudaMemcpy(data+(firstTimeSample+1)*host_nz*host_nx, dev_p0[iGpu]+1*host_nz*host_nx, (lastTimeSample-firstTimeSample-1)*host_nz*host_nx*sizeof(float), cudaMemcpyDeviceToHost));
	}
}
void waveEquationAcousticAdjGpu(float *model,float *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample){

	// Set GPU
  cudaSetDevice(iGpuId);

	//copy data to gpu
	cuda_call(cudaMemcpy(dev_p1[iGpu], data+firstTimeSample*host_nz*host_nx, (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*sizeof(float), cudaMemcpyHostToDevice));

	//copy model to gpu
	cuda_call(cudaMemcpy(dev_p0[iGpu], model+firstTimeSample*host_nz*host_nx, (lastTimeSample-firstTimeSample+1)*host_nz*host_nx*sizeof(float), cudaMemcpyHostToDevice));

	//call adj gpu kernel
	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	int nblockz = ((lastTimeSample-firstTimeSample+1)+BLOCK_SIZE) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky, nblockz);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

	kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p1[iGpu],firstTimeSample,lastTimeSample));
	kernel_exec(ker_we_adj<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_slsqDt2[iGpu],dev_cosDamp[iGpu],firstTimeSample,lastTimeSample));

	//copy model from gpu
	//first, last, and other gpus have different copies
	if(firstTimeSample==0 && lastTimeSample==host_nts-1) {
		//include first and last sample in block
		cuda_call(cudaMemcpy(model, dev_p0[iGpu], host_nts*host_nz*host_nx*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else if(firstTimeSample==0){
		//exclude last sample | include first sample in block
		cuda_call(cudaMemcpy(model, dev_p0[iGpu], (lastTimeSample-firstTimeSample)*host_nz*host_nx*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else if(lastTimeSample==host_nts-1){
		//exclude first sample | include last sample in block
		cuda_call(cudaMemcpy(model+(firstTimeSample+1)*host_nz*host_nx, dev_p0[iGpu]+1*host_nz*host_nx, (lastTimeSample-firstTimeSample)*host_nz*host_nx*sizeof(float), cudaMemcpyDeviceToHost));
	}
	else{
		//exclude first and last sample in block
		cuda_call(cudaMemcpy(model+(firstTimeSample+1)*host_nz*host_nx, dev_p0[iGpu]+1*host_nz*host_nx, (lastTimeSample-firstTimeSample-1)*host_nz*host_nx*sizeof(float), cudaMemcpyDeviceToHost));
	}
}

// void deallocateWaveEquationAcousticGpu(){
//   // Deallocate scaled elastic params
//   cuda_call(cudaFree(dev_rhox));
//   cuda_call(cudaFree(dev_rhoz));
//   cuda_call(cudaFree(dev_lamb2Muw));
//   cuda_call(cudaFree(dev_lamb));
//   cuda_call(cudaFree(dev_muxzw));
//
//   // Deallocate wavefields
// 	cuda_call(cudaFree(dev_p0));
//   cuda_call(cudaFree(dev_p1));


// check gpu info
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo){

	int nDevice, driver;
	cudaGetDeviceCount(&nDevice);

	if (info == 1){

		std::cout << " " << std::endl;
		std::cout << "*******************************************************************" << std::endl;
		std::cout << "************************ INFO FOR GPU# " << deviceNumberInfo << " *************************" << std::endl;
		std::cout << "*******************************************************************" << std::endl;

		// Number of devices
		std::cout << "Number of requested GPUs: " << nGpu << std::endl;
		std::cout << "Number of available GPUs: " << nDevice << std::endl;

		// Driver version
		std::cout << "Cuda driver version: " << cudaDriverGetVersion(&driver) << std::endl; // Driver

		// Get properties
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop,deviceNumberInfo);

		// Display
		std::cout << "Name: " << dprop.name << std::endl;
		std::cout << "Total global memory: " << dprop.totalGlobalMem/(1024*1024*1024) << " [GB] " << std::endl;
		std::cout << "Shared memory per block: " << dprop.sharedMemPerBlock/1024 << " [kB]" << std::endl;
		std::cout << "Number of register per block: " << dprop.regsPerBlock << std::endl;
		std::cout << "Warp size: " << dprop.warpSize << " [threads]" << std::endl;
		std::cout << "Maximum pitch allowed for memory copies in bytes: " << dprop.memPitch/(1024*1024*1024) << " [GB]" << std::endl;
		std::cout << "Maximum threads per block: " << dprop.maxThreadsPerBlock << std::endl;
		std::cout << "Maximum block dimensions: " << "(" << dprop.maxThreadsDim[0] << ", " << dprop.maxThreadsDim[1] << ", " << dprop.maxThreadsDim[2] << ")" << std::endl;
		std::cout << "Maximum grid dimensions: " << "(" << dprop.maxGridSize[0] << ", " << dprop.maxGridSize[1] << ", " << dprop.maxGridSize[2] << ")" << std::endl;
		std::cout << "Total constant memory: " << dprop.totalConstMem/1024 << " [kB]" << std::endl;
		std::cout << "Number of streaming multiprocessors on device: " << dprop.multiProcessorCount << std::endl;
		if (dprop.deviceOverlap == 1) {std::cout << "Device can simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.deviceOverlap != 1) {std::cout << "Device cannot simultaneously perform a cudaMemcpy() and kernel execution" << std::endl;}
		if (dprop.canMapHostMemory == 1) { std::cout << "Device can map host memory" << std::endl; }
		if (dprop.canMapHostMemory != 1) { std::cout << "Device cannot map host memory" << std::endl; }
		if (dprop.concurrentKernels == 1) {std::cout << "Device can support concurrent kernel" << std::endl;}
		if (dprop.concurrentKernels != 1) {std::cout << "Device cannot support concurrent kernel execution" << std::endl;}

		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << " " << std::endl;
	}

  	if (nGpu<nDevice+1) {return true;}
  	else {std::cout << "Number of requested GPU greater than available GPUs" << std::endl; return false;}
}

float getTotalGlobalMem(int nGpu, int info, int deviceNumberInfo){
	int nDevice, driver;
	cudaGetDeviceCount(&nDevice);
	// Get properties
	cudaDeviceProp dprop;
	cudaGetDeviceProperties(&dprop,deviceNumberInfo);
	return dprop.totalGlobalMem/(1024*1024*1024);
}


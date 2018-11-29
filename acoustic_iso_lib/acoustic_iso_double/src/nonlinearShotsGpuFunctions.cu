#include <cstring>
#include <iostream>
#include "nonlinearShotsGpuFunctions.h"
#include "varDeclare.h"
#include "kernelsGpu.cu"
#include "cudaErrors.cu"
#include <vector>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <assert.h>

/****************************************************************************************/
/******************************* Set GPU propagation parameters *************************/
/****************************************************************************************/
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo){

	int nDevice, driver;
	cudaGetDeviceCount(&nDevice);

	if (info == 1){

		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "---------------------------- INFO FOR GPU# " << deviceNumberInfo << " ----------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;

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
void initNonlinearGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, double alphaCos, int nGpu, int iGpu){

	// Set GPU
	cudaSetDevice(iGpu);

	// Host variables
	host_nz = nz;
	host_nx = nx;
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;

	/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpu == 0) {

		// Time slices for FD stepping
		dev_p0 = new double*[nGpu];
		dev_p1 = new double*[nGpu];
		dev_temp1 = new double*[nGpu];

		// Data and model
		dev_modelRegDtw = new double*[nGpu];
		dev_dataRegDts = new double*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new int*[nGpu];
		dev_receiversPositionReg = new int*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new double*[nGpu];

	}

	/**************************** COMPUTE LAPLACIAN COEFFICIENTS ************************/
	double zCoeff[COEFF_SIZE];
	double xCoeff[COEFF_SIZE];

	zCoeff[0] = -2.927222222 / (dz * dz);
  	zCoeff[1] = 1.666666667 / (dz * dz);
  	zCoeff[2] = -0.238095238 / (dz * dz);
  	zCoeff[3] = 0.039682539 / (dz * dz);
  	zCoeff[4] = -0.004960317 / (dz * dz);
  	zCoeff[5] = 0.000317460 / (dz * dz);

  	xCoeff[0] = -2.927222222 / (dx * dx);
  	xCoeff[1] = 1.666666667 / (dx * dx);
  	xCoeff[2] = -0.238095238 / (dx * dx);
  	xCoeff[3] = 0.039682539 / (dx * dx);
  	xCoeff[4] = -0.004960317 / (dx * dx);
  	xCoeff[5] = 0.000317460 / (dx * dx);

	/**************************** COMPUTE TIME-INTERPOLATION FILTER *********************/
	// Time interpolation filter length/half length
	int hInterpFilter = host_sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		std::cout << "**** ERROR: Subsampling parameter too high ****" << std::endl;
		assert (1==2);
	}

	// Allocate and fill interpolation filter
	double interpFilter[nInterpFilter];
	for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
		interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
		interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
		interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
		interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(double(host_ntw)/double(host_nts)));
	}

	/************************* COMPUTE COSINE DAMPING COEFFICIENTS **********************/
	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR: Padding value is too high ****" << std::endl;
		assert (1==2);
	}
	double cosDampingCoeff[minPad];

	// Cosine padding
	for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
		double arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		cosDampingCoeff[iFilter-FAT] = arg;
	}

	/**************************** COPY TO CONSTANT MEMORY *******************************/
	// Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_zCoeff, zCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice)); // Copy Laplacian coefficients to device
	cuda_call(cudaMemcpyToSymbol(dev_xCoeff, xCoeff, COEFF_SIZE*sizeof(double), 0, cudaMemcpyHostToDevice));

	// Time interpolation filter
	cuda_call(cudaMemcpyToSymbol(dev_nInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
	cuda_call(cudaMemcpyToSymbol(dev_hInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
	cuda_call(cudaMemcpyToSymbol(dev_interpFilter, interpFilter, nInterpFilter*sizeof(double), 0, cudaMemcpyHostToDevice)); // Filter

	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(double), 0, cudaMemcpyHostToDevice)); // Array for damping
	cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(double), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula
	cuda_call(cudaMemcpyToSymbol(dev_minPad, &minPad, sizeof(int), 0, cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nz, &nz, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

}
void allocateNonlinearGpu(double *vel2Dtw2, int iGpu){

	// Get GPU number
	cudaSetDevice(iGpu);

	// Scaled velocity
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nz*host_nx*sizeof(double))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate time slices on device
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nz*host_nx*sizeof(double))); // Allocate time slices on device (for the stepper)
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nz*host_nx*sizeof(double)));

}
void deallocateNonlinearGpu(int iGpu){
	cudaSetDevice(iGpu); // Set device number on GPU cluster
  cuda_call(cudaFree(dev_vel2Dtw2[iGpu])); // Deallocate scaled velocity
	cuda_call(cudaFree(dev_p0[iGpu]));
  cuda_call(cudaFree(dev_p1[iGpu]));
}

/****************************************************************************************/
/******************************* Nonlinear forward propagation **************************/
/****************************************************************************************/
void propShotsFwdGpu(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpu);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Model
	cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate output on device
	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize output on device

	// Time slices
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Extraction grid size
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Start propagation
	for (int its = 0; its < host_nts-1; its++){

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract and interpolate data
			kernel_exec(recordInterpData<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate all slices
  cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
  cuda_call(cudaFree(dev_dataRegDts[iGpu]));
  cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
  cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}
void propShotsFwdGpuWavefield(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpu);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Model
  cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate input on device
	cuda_call(cudaMemcpy(dev_modelRegDtw[iGpu], modelRegDtw, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy input signals on device

	// Data
  cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate output on device
  cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize output on device

 	// Wavefield
	cuda_call(cudaMalloc((void**) &dev_wavefieldDts, host_nz*host_nx*host_nts*sizeof(double))); // Allocate on device
	cuda_call(cudaMemset(dev_wavefieldDts, 0, host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

	// Time slices
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double))); // Initialize time slices on device
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Laplacian grid and blocks
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Extraction grid size
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Start propagation
	for (int its = 0; its < host_nts-1; its++){

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_modelRegDtw[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_wavefieldDts, dev_p0[iGpu], its, it2));

			// Extract and interpolate data
			kernel_exec(recordInterpData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(wavefieldDts, dev_wavefieldDts, host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate all slices
  cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
  cuda_call(cudaFree(dev_dataRegDts[iGpu]));
  cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
  cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_wavefieldDts));

}

/****************************************************************************************/
/******************************* Nonlinear adjoint propagation **************************/
/****************************************************************************************/
void propShotsAdjGpu(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpu);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Model
  cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate model on device
  cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(double))); // Initialize model on device

	// Data
  cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy data on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Start propagation
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward in time
			kernel_exec(stepAdjGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract model
			kernel_exec(recordSource<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate all slices
  cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
  cuda_call(cudaFree(dev_dataRegDts[iGpu]));
  cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
  cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}
void propShotsAdjGpuWavefield(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu) {

	// Set device number on GPU cluster
	cudaSetDevice(iGpu);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Model
  cuda_call(cudaMalloc((void**) &dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate model on device
	cuda_call(cudaMemset(dev_modelRegDtw[iGpu], 0, nSourcesReg*host_ntw*sizeof(double))); // Initialize model on device

	// Data
  cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy data on device

 	// Wavefield
	cuda_call(cudaMalloc((void**) &dev_wavefieldDts, host_nz*host_nx*host_nts*sizeof(double))); // Allocate on device
	cuda_call(cudaMemset(dev_wavefieldDts, 0, host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

	// Initialize time slices on device
	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Grid and block dimensions for stepper
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	// Grid and block dimensions for data injection
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;

	// Start propagation
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward in time
			kernel_exec(stepAdjGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and save wavefield on device (the wavefield is not scaled)
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_wavefieldDts, dev_p0[iGpu], its, it2));

			// Extract model
			kernel_exec(recordSource<<<1, nSourcesReg>>>(dev_p0[iGpu], dev_modelRegDtw[iGpu], itw, dev_sourcesPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(modelRegDtw, dev_modelRegDtw[iGpu], nSourcesReg*host_ntw*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(wavefieldDts, dev_wavefieldDts, host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Deallocate all slices
  cuda_call(cudaFree(dev_modelRegDtw[iGpu]));
  cuda_call(cudaFree(dev_dataRegDts[iGpu]));
  cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
  cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	cuda_call(cudaFree(dev_wavefieldDts));

}

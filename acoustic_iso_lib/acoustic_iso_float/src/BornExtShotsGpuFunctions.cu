#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include "BornExtShotsGpuFunctions.h"
#include "kernelsGpu.cu"
#include "cudaErrors.cu"
#include "varDeclare.h"
#include <stdio.h>
#include <assert.h>

/******************************************************************************/
/*********************** Set GPU propagation parameters ***********************/
/******************************************************************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo){

	int nDevice, driver;
	cudaGetDeviceCount(&nDevice);

	if (info == 1){

		std::cout << " " << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;
		std::cout << "---------------------------- INFO FOR GPU# " << deviceNumberInfo << " ----------------------" << std::endl;
		std::cout << "-------------------------------------------------------------------" << std::endl;

		// Number of devices
		std::cout << "Number of requested GPUs: " << gpuList.size() << std::endl;
		std::cout << "Number of available GPUs: " << nDevice << std::endl;
		std::cout << "Id of requested GPUs: ";
		for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
			if (iGpu<gpuList.size()-1){std::cout << gpuList[iGpu] << ", ";}
 			else{ std::cout << gpuList[iGpu] << std::endl;}
		}

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

	// Check that the number of requested GPU is less or equal to the total number of available GPUs
	if (gpuList.size()>nDevice) {
		std::cout << "**** ERROR [getGpuInfo]: Number of requested GPU greater than available GPUs ****" << std::endl;
		return false;
	}

	// Check that the GPU numbers in the list are between 0 and nGpu-1
	for (int iGpu=0; iGpu<gpuList.size(); iGpu++){
		if (gpuList[iGpu]<0 || gpuList[iGpu]>nDevice-1){
			std::cout << "**** ERROR [getGpuInfo]: One of the element of the GPU Id list is not a valid GPU Id number ****" << std::endl;
			return false;
		}
	}

	return true;
}
void initBornExtGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nExt, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU number
	cudaSetDevice(iGpuId);

	host_nz = nz;
	host_nx = nx;
	host_dz = dz;
	host_dx = dx;
	host_nExt = nExt;
	host_hExt = (nExt-1)/2;
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;

	/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new float*[nGpu];
		dev_p1 = new float*[nGpu];
		dev_temp1 = new float*[nGpu];

		dev_ssLeft = new float*[nGpu];
		dev_ssRight = new float*[nGpu];
		dev_ssTemp1 = new float*[nGpu];

		// Data
		dev_dataRegDts = new float*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new int*[nGpu];
		dev_receiversPositionReg = new int*[nGpu];

		// Sources signal
		dev_sourcesSignals = new float*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new float*[nGpu];

		// Reflectivity scaling
		dev_reflectivityScale = new float*[nGpu];

		// Reflectivity
		dev_modelBornExt = new float*[nGpu];

		// Source and secondary wavefields
		dev_BornSrcWavefield = new float*[nGpu];

	}

	/**************************** COMPUTE LAPLACIAN COEFFICIENTS ************************/
	float zCoeff[COEFF_SIZE];
	float xCoeff[COEFF_SIZE];

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
	int hInterpFilter = sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>=SUB_MAX){
		std::cout << "**** ERROR: Subsampling parameter is too high ****" << std::endl;
		throw std::runtime_error("");
	}

	// Allocate and fill interpolation filter
	float interpFilter[nInterpFilter];
	for (int iFilter = 0; iFilter < hInterpFilter; iFilter++){
		interpFilter[iFilter] = 1.0 - 1.0 * iFilter/host_sub;
		interpFilter[iFilter+hInterpFilter] = 1.0 - interpFilter[iFilter];
		interpFilter[iFilter] = interpFilter[iFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
		interpFilter[iFilter+hInterpFilter] = interpFilter[iFilter+hInterpFilter] * (1.0 / sqrt(float(host_ntw)/float(host_nts)));
	}

	/************************* COMPUTE COSINE DAMPING COEFFICIENTS **********************/
	// Check padding is smaller than maximum allowed
	if (minPad>=PAD_MAX){
		std::cout << "**** ERROR: Padding value is too high ****" << std::endl;
		throw std::runtime_error("");
	}
	float cosDampingCoeff[minPad];

	// Cosine padding
	for (int iFilter=FAT; iFilter<FAT+minPad; iFilter++){
		float arg = M_PI / (1.0 * minPad) * 1.0 * (minPad-iFilter+FAT);
		arg = alphaCos + (1.0-alphaCos) * cos(arg);
		cosDampingCoeff[iFilter-FAT] = arg;
	}

	// Check that the block size is consistent between parfile and "varDeclare.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR: Block size for time stepper is not consistent with parfile ****" << std::endl;
		throw std::runtime_error("");
	}

	/**************************** COPY TO CONSTANT MEMORY *******************************/
	// Laplacian coefficients
	cuda_call(cudaMemcpyToSymbol(dev_zCoeff, zCoeff, COEFF_SIZE*sizeof(float), 0, cudaMemcpyHostToDevice)); // Copy Laplacian coefficients to device
	cuda_call(cudaMemcpyToSymbol(dev_xCoeff, xCoeff, COEFF_SIZE*sizeof(float), 0, cudaMemcpyHostToDevice));

	// Time interpolation filter
	cuda_call(cudaMemcpyToSymbol(dev_nInterpFilter, &nInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter length
	cuda_call(cudaMemcpyToSymbol(dev_hInterpFilter, &hInterpFilter, sizeof(int), 0, cudaMemcpyHostToDevice)); // Filter half-length
	cuda_call(cudaMemcpyToSymbol(dev_interpFilter, interpFilter, nInterpFilter*sizeof(float), 0, cudaMemcpyHostToDevice)); // Filter

	// Cosine damping parameters
	cuda_call(cudaMemcpyToSymbol(dev_cosDampingCoeff, &cosDampingCoeff, minPad*sizeof(float), 0, cudaMemcpyHostToDevice)); // Array for damping
	cuda_call(cudaMemcpyToSymbol(dev_alphaCos, &alphaCos, sizeof(float), 0, cudaMemcpyHostToDevice)); // Coefficient in the damping formula
	cuda_call(cudaMemcpyToSymbol(dev_minPad, &minPad, sizeof(int), 0, cudaMemcpyHostToDevice)); // min (zPadMinus, zPadPlus, xPadMinus, xPadPlus)

	// FD parameters
	cuda_call(cudaMemcpyToSymbol(dev_nz, &nz, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy model size to device
	cuda_call(cudaMemcpyToSymbol(dev_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_nts, &nts, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device
	cuda_call(cudaMemcpyToSymbol(dev_sub, &sub, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_ntw, &host_ntw, sizeof(int), 0, cudaMemcpyHostToDevice)); // Copy number of coarse time parameters to device

	// Extension parameters
	cuda_call(cudaMemcpyToSymbol(dev_nExt, &host_nExt, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_hExt, &host_hExt, sizeof(int), 0, cudaMemcpyHostToDevice));

}
void allocateBornExtShotsGpu(float *vel2Dtw2, float *reflectivityScale, int iGpu, int iGpuId){

	// Set GPU number
	cudaSetDevice(iGpuId);

	// Reflectivity scale
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nz*host_nx*sizeof(float))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nz*host_nx*sizeof(float), cudaMemcpyHostToDevice)); //

	// Scaled velocity
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nz*host_nx*sizeof(float))); // Allocate scaling for reflectivity
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nz*host_nx*sizeof(float), cudaMemcpyHostToDevice)); //

	// Allocate time slices
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nz*host_nx*sizeof(float)));
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMalloc((void**) &dev_ssLeft[iGpu], host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMalloc((void**) &dev_ssRight[iGpu], host_nz*host_nx*sizeof(float)));

	// Allocate non-extended model
	cuda_call(cudaMalloc((void**) &dev_modelBornExt[iGpu], host_nz*host_nx*host_nExt*sizeof(float)));

	// Allocate source wavefield
	cuda_call(cudaMalloc((void**) &dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(float))); // We store the source wavefield

}
void deallocateBornExtShotsGpu(int iGpu, int iGpuId){

 		// Set device number on GPU cluster
		cudaSetDevice(iGpuId);

		// Deallocate all the shit
    	cuda_call(cudaFree(dev_vel2Dtw2[iGpu]));
    	cuda_call(cudaFree(dev_reflectivityScale[iGpu]));
		cuda_call(cudaFree(dev_p0[iGpu]));
    	cuda_call(cudaFree(dev_p1[iGpu]));
		cuda_call(cudaFree(dev_ssLeft[iGpu]));
		cuda_call(cudaFree(dev_ssRight[iGpu]));
		cuda_call(cudaFree(dev_BornSrcWavefield[iGpu]));
		cuda_call(cudaFree(dev_modelBornExt[iGpu]));

}

/******************************************************************************/
/**************************** Born extended forward ***************************/
/******************************************************************************/

/********************************** Normal ************************************/
// Time-lags
void BornTimeShotsFwdGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int sloth, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	/************************** Scattered wavefield computation *************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and copy model
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyHostToDevice)); // Copy extended model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	// Apply both scalings to reflectivity:
	// First: -2.0 * 1/v^3 * v^2 * dtw^2
	if (sloth==0){
		kernel_exec(scaleReflectivityExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));
	} else {
		kernel_exec(scaleReflectivitySlothExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu]));
	}

	// Compute secondary source for first coarse time index (its = 0)
	int its = 0;
	int iExtMin, iExtMax;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Add 1 for the strict inequality in the "for loop"
	kernel_exec(imagingTimeFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssLeft[iGpu], dev_BornSrcWavefield[iGpu], its, iExtMin, iExtMax)); // Apply extended FWD imaging condition

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt) + host_hExt; // Lower bound for extended index
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for extended index
		kernel_exec(imagingTimeFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1, iExtMin, iExtMax)); // Apply time-extended FWD imaging condition

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract data
			kernel_exec(recordInterpData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source after second time derivative
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
}

void BornTimeShotsFwdGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(srcWavefieldDts, dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/************************** Scattered wavefield computation *************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and copy model
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	// Allocate and initialize scattered wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(float))); // Initialize wavefield on device

	// Apply both scalings to reflectivity:
	// First: -2.0 * 1/v^3 * v^2 * dtw^2
	kernel_exec(scaleReflectivityExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Compute secondary source for first coarse time index (its = 0)
	int its = 0;
	int iExtMin, iExtMax;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Add 1 for the strict inequality in the "for loop"
	kernel_exec(imagingTimeFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssLeft[iGpu], dev_BornSrcWavefield[iGpu], its, iExtMin, iExtMax)); // Apply extended FWD imaging condition

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt) + host_hExt; // Lower bound for extended index
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for extended index
		kernel_exec(imagingTimeFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1, iExtMin, iExtMax)); // Apply time-extended FWD imaging condition

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Record wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_p0[iGpu], its, it2));

			// Extract data
			kernel_exec(recordInterpData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source after second time derivative
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy scattered wavefield back to host
	cuda_call(cudaMemcpy(scatWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));
}

// Subsurface offsets
void BornOffsetShotsFwdGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	/************************** Scattered wavefield computation *************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and copy model
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	// Apply first scaling to reflectivity: 2.0 * 1/v^3 coming from the linearization of the wave equation
	kernel_exec(scaleReflectivityLinExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu]));

	// Compute secondary source for first coarse time index (its = 0)
	// Apply extended fwd imaging condition with subsurface offset extension
	kernel_exec(imagingOffsetFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssLeft[iGpu], dev_BornSrcWavefield[iGpu], 0));

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_vel2Dtw2[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		kernel_exec(imagingOffsetFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1)); // Apply time-extended FWD imaging condition

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract data
			kernel_exec(recordInterpData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source after second time derivative
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void BornOffsetShotsFwdGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(srcWavefieldDts, dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/************************** Scattered wavefield computation *************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and copy model
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	// Allocate and initialize scattered wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(float))); // Initialize wavefield on device

	// Apply first scaling to reflectivity: 2.0 * 1/v^3 coming from the linearization of the wave equation
	kernel_exec(scaleReflectivityLinExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu]));

	// Compute secondary source for first coarse time index (its = 0)
	// Apply extended fwd imaging condition with subsurface offset extension
	kernel_exec(imagingOffsetFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssLeft[iGpu], dev_BornSrcWavefield[iGpu], 0));

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_vel2Dtw2[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		kernel_exec(imagingOffsetFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1)); // Apply time-extended FWD imaging condition

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Record wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_p0[iGpu], its, it2));

			// Extract data
			kernel_exec(recordInterpData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source after second time derivative
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy scattered wavefield back to host
	cuda_call(cudaMemcpy(scatWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));
}

/********************************** Free surface ******************************/
// Time-lags
void BornTimeShotsFwdFsGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	/************************** Scattered wavefield computation *************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and copy model
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyHostToDevice)); // Copy extended model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	// Apply both scalings to reflectivity:
	// First: -2.0 * 1/v^3 * v^2 * dtw^2
	kernel_exec(scaleReflectivityExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Compute secondary source for first coarse time index (its = 0)
	int its = 0;
	int iExtMin, iExtMax;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Add 1 for the strict inequality in the "for loop"
	kernel_exec(imagingTimeFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssLeft[iGpu], dev_BornSrcWavefield[iGpu], its, iExtMin, iExtMax)); // Apply extended FWD imaging condition

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt) + host_hExt; // Lower bound for extended index
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for extended index
		kernel_exec(imagingTimeFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1, iExtMin, iExtMax)); // Apply time-extended FWD imaging condition

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract data
			kernel_exec(recordInterpData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source after second time derivative
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
}

void BornTimeShotsFwdFsGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(srcWavefieldDts, dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/************************** Scattered wavefield computation *************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and copy model
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	// Allocate and initialize scattered wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(float))); // Initialize wavefield on device

	// Apply both scalings to reflectivity:
	// First: -2.0 * 1/v^3 * v^2 * dtw^2
	kernel_exec(scaleReflectivityExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Compute secondary source for first coarse time index (its = 0)
	int its = 0;
	int iExtMin, iExtMax;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Add 1 for the strict inequality in the "for loop"
	kernel_exec(imagingTimeFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssLeft[iGpu], dev_BornSrcWavefield[iGpu], its, iExtMin, iExtMax)); // Apply extended FWD imaging condition

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt) + host_hExt; // Lower bound for extended index
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for extended index
		kernel_exec(imagingTimeFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1, iExtMin, iExtMax)); // Apply time-extended FWD imaging condition

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Record wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_p0[iGpu], its, it2));

			// Extract data
			kernel_exec(recordInterpData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source after second time derivative
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy scattered wavefield back to host
	cuda_call(cudaMemcpy(scatWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));
}

// Subsurface offsets
void BornOffsetShotsFwdFsGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	/************************** Scattered wavefield computation *************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and copy model
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	// Apply first scaling to reflectivity: 2.0 * 1/v^3 coming from the linearization of the wave equation
	kernel_exec(scaleReflectivityLinExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu]));

	// Compute secondary source for first coarse time index (its = 0)
	// Apply extended fwd imaging condition with subsurface offset extension
	kernel_exec(imagingOffsetFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssLeft[iGpu], dev_BornSrcWavefield[iGpu], 0));

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_vel2Dtw2[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		kernel_exec(imagingOffsetFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1)); // Apply time-extended FWD imaging condition

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract data
			kernel_exec(recordInterpData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source after second time derivative
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void BornOffsetShotsFwdFsGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(srcWavefieldDts, dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/************************** Scattered wavefield computation *************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and copy model
	cuda_call(cudaMemcpy(dev_modelBornExt[iGpu], model, host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(float))); // Initialize data on device

	// Allocate and initialize scattered wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(float))); // Initialize wavefield on device

	// Apply first scaling to reflectivity: 2.0 * 1/v^3 coming from the linearization of the wave equation
	kernel_exec(scaleReflectivityLinExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu]));

	// Compute secondary source for first coarse time index (its = 0)
	// Apply extended fwd imaging condition with subsurface offset extension
	kernel_exec(imagingOffsetFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssLeft[iGpu], dev_BornSrcWavefield[iGpu], 0));

	// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
	kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_vel2Dtw2[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		kernel_exec(imagingOffsetFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1)); // Apply time-extended FWD imaging condition

		// Apply second scaling to secondary source: v^2 * dtw^2 coming from the finite difference scheme
		kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Record wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_p0[iGpu], its, it2));

			// Extract data
			kernel_exec(recordInterpData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source after second time derivative
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy scattered wavefield back to host
	cuda_call(cudaMemcpy(scatWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));
}

/******************************************************************************/
/**************************** Born extended adjoint ***************************/
/******************************************************************************/

/********************************** Normal ************************************/
// Time-lags
void BornTimeShotsAdjGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *recWavefieldDts, int sloth, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	/************************** Receiver wavefield computation **************************/

	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Model
  	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nz*host_nx*host_nExt*sizeof(float))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Declare min/max index for extended imaging condition
	int iExtMin, iExtMax;

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			extractInterpAdjointWavefield<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers for time slices at fine time-sampling
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply extended imaging condition for its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for time-lag index
  		kernel_exec(imagingTimeAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1, iExtMin, iExtMax));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Apply extended imaging condition for its=0
	// Compute time-extension bounds
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1;
	kernel_exec(imagingTimeAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its, iExtMin, iExtMax)); // Imaging kernel for its=0

  	// Scale model for finite-difference and secondary source coefficient
	if (sloth==0){
		kernel_exec(scaleReflectivityExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));
	} else {
		kernel_exec(scaleReflectivitySlothExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_vel2Dtw2[iGpu]));
	}

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void BornTimeShotsAdjGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *recWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(srcWavefieldDts, dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/************************** Receiver wavefield computation **************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and initialize receiver wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(float))); // Initialize wavefield on device

	// Model
  	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nz*host_nx*host_nExt*sizeof(float)));

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

	// Declare min/max index for extended imaging condition
	int iExtMin, iExtMax;

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			extractInterpAdjointWavefield<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers for time slices at fine time-sampling
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Record and scale receiver wavefield at coarse sampling for its+1
		kernel_exec(recordScaleWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], its+1, dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

		// Apply imaging condition for its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for time-lag index
  		kernel_exec(imagingTimeAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1, iExtMin, iExtMax));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Save receiver wavefield at its=0
	kernel_exec(recordScaleWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], 0, dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	/**************************** Extended imaging condition ****************************/
	// Compute time-extension bounds
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for time-lag index
	kernel_exec(imagingTimeAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its, iExtMin, iExtMax)); // Imaging kernel for its=0

  	// Scale model for finite-difference and secondary source coefficient
	// It's better to apply it once and for all than at every time-steps
	kernel_exec(scaleReflectivityExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy receiver wavefield back to host
	cuda_call(cudaMemcpy(recWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));
}

// Subsurface offsets
void BornOffsetShotsAdjGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *recWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	/************************** Receiver wavefield computation **************************/

	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Model
  	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nz*host_nx*host_nExt*sizeof(float))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			extractInterpAdjointWavefield<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers for time slices at fine time-sampling
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}

		// Scale the receiver wavefield by v^2 * dtw^2
		kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

		// Apply imaging condition for its+1
  		kernel_exec(imagingOffsetAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Scale the receiver wavefield by v^2 * dtw^2
	kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

	// Subsurface offset extended imaging condition
	kernel_exec(imagingOffsetAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], 0)); // Imaging kernel for its=0

  	// Scale model by 2/v^3
	kernel_exec(scaleReflectivityLinExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu]));

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void BornOffsetShotsAdjGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *recWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(srcWavefieldDts, dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/************************** Receiver wavefield computation **************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and initialize receiver wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(float))); // Initialize wavefield on device

	// Model
  	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nz*host_nx*host_nExt*sizeof(float)));

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			extractInterpAdjointWavefield<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers for time slices at fine time-sampling
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Scale the receiver wavefield by v^2 * dtw^2
		kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

		// Record and scale receiver wavefield at coarse sampling for its+1
		kernel_exec(recordWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], its+1));

		// Apply imaging condition for its+1
  		kernel_exec(imagingOffsetAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Scale the receiver wavefield by v^2 * dtw^2
	kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

	// Save receiver wavefield at its = 0
	kernel_exec(recordWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], 0));

	// Subsurface offset extended imaging condition for its = 0
	kernel_exec(imagingOffsetAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], 0));

	// Scale model for finite-difference and secondary source coefficient
	kernel_exec(scaleReflectivityLinExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu]));

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy receiver wavefield back to host
	cuda_call(cudaMemcpy(recWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));
}

/********************************** Free surface ******************************/
// Time-lags
void BornTimeShotsAdjFsGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *recWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	/************************** Receiver wavefield computation **************************/

	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Model
  	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nz*host_nx*host_nExt*sizeof(float))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float)));
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice));

	// Declare min/max index for extended imaging condition
	int iExtMin, iExtMax;

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjFsGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			extractInterpAdjointWavefield<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers for time slices at fine time-sampling
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply extended imaging condition for its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for time-lag index
  		kernel_exec(imagingTimeAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1, iExtMin, iExtMax));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Apply extended imaging condition for its=0
	// Compute time-extension bounds
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1;
	kernel_exec(imagingTimeAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its, iExtMin, iExtMax)); // Imaging kernel for its=0

  	// Scale model for finite-difference and secondary source coefficient
	kernel_exec(scaleReflectivityExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void BornTimeShotsAdjFsGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *recWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(srcWavefieldDts, dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/************************** Receiver wavefield computation **************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and initialize receiver wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(float))); // Initialize wavefield on device

	// Model
  	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nz*host_nx*host_nExt*sizeof(float)));

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

	// Declare min/max index for extended imaging condition
	int iExtMin, iExtMax;

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjFsGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			extractInterpAdjointWavefield<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers for time slices at fine time-sampling
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Record and scale receiver wavefield at coarse sampling for its+1
		kernel_exec(recordScaleWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], its+1, dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

		// Apply imaging condition for its+1
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for time-lag index
  		kernel_exec(imagingTimeAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1, iExtMin, iExtMax));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Save receiver wavefield at its=0
	kernel_exec(recordScaleWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], 0, dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	/**************************** Extended imaging condition ****************************/
	// Compute time-extension bounds
	int its = 0;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for time-lag index
	kernel_exec(imagingTimeAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its, iExtMin, iExtMax)); // Imaging kernel for its=0

  	// Scale model for finite-difference and secondary source coefficient
	// It's better to apply it once and for all than at every time-steps
	kernel_exec(scaleReflectivityExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy receiver wavefield back to host
	cuda_call(cudaMemcpy(recWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));
}

// Subsurface offsets
void BornOffsetShotsAdjFsGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *recWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	/************************** Receiver wavefield computation **************************/

	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Model
  	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nz*host_nx*host_nExt*sizeof(float))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjFsGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			extractInterpAdjointWavefield<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers for time slices at fine time-sampling
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}

		// Scale the receiver wavefield by v^2 * dtw^2
		kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

		// Apply imaging condition for its+1
  		kernel_exec(imagingOffsetAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Scale the receiver wavefield by v^2 * dtw^2
	kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

	// Subsurface offset extended imaging condition
	kernel_exec(imagingOffsetAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], 0)); // Imaging kernel for its=0

  	// Scale model by 2/v^3
	kernel_exec(scaleReflectivityLinExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu]));

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}

void BornOffsetShotsAdjFsGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *recWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(float))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(float), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(float)));

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

   	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

   	// Extended blocks/threads
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);

	/************************** Source wavefield computation ****************************/
	for (int its = 0; its < host_nts-1; its++){
		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Compute fine time-step index
			int itw = its * host_sub + it2;

			// Apply free surface condition for Laplacian
			kernel_exec(setFsConditionFwdGpu<<<nBlockX, BLOCK_SIZE>>>(dev_p1[iGpu]));

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject source
			kernel_exec(injectSource<<<1, nSourcesReg>>>(dev_sourcesSignals[iGpu], dev_p0[iGpu], itw-1, dev_sourcesPositionReg[iGpu]));

			// Damp wavefields
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract wavefield
			kernel_exec(interpWavefield<<<dimGrid, dimBlock>>>(dev_BornSrcWavefield[iGpu], dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}
	}

	// Copy wavefield back to host
	cuda_call(cudaMemcpy(srcWavefieldDts, dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/************************** Receiver wavefield computation **************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(float)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(float)));

	// Allocate and initialize receiver wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(float))); // Initialize wavefield on device

	// Model
  	cuda_call(cudaMemset(dev_modelBornExt[iGpu], 0, host_nz*host_nx*host_nExt*sizeof(float)));

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(float))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(float), cudaMemcpyHostToDevice)); // Copy data on device

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjFsGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nBlockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Damp wavefield
			kernel_exec(dampCosineEdgeFs<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and record time slices of receiver wavefield at coarse sampling (no scaling applied yet)
			extractInterpAdjointWavefield<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2);

			// Switch pointers for time slices at fine time-sampling
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Scale the receiver wavefield by v^2 * dtw^2
		kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

		// Record and scale receiver wavefield at coarse sampling for its+1
		kernel_exec(recordWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], its+1));

		// Apply imaging condition for its+1
  		kernel_exec(imagingOffsetAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(float))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Scale the receiver wavefield by v^2 * dtw^2
	kernel_exec(scaleSecondarySourceFd<<<dimGrid, dimBlock>>>(dev_ssRight[iGpu], dev_vel2Dtw2[iGpu]));

	// Save receiver wavefield at its = 0
	kernel_exec(recordWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], 0));

	// Subsurface offset extended imaging condition for its = 0
	kernel_exec(imagingOffsetAdjGpu<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], 0));

	// Scale model for finite-difference and secondary source coefficient
	kernel_exec(scaleReflectivityLinExt<<<dimGridExt, dimBlockExt>>>(dev_modelBornExt[iGpu], dev_reflectivityScale[iGpu]));

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBornExt[iGpu], host_nz*host_nx*host_nExt*sizeof(float), cudaMemcpyDeviceToHost));

	// Copy receiver wavefield back to host
	cuda_call(cudaMemcpy(recWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(float), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));
}

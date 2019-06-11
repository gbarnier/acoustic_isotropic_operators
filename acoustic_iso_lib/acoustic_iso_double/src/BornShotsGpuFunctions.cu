#include "BornShotsGpuFunctions.h"
#include <iostream>
#include "varDeclare.h"
#include <vector>
#include <algorithm>
#include <math.h>
#include "kernelsGpu.cu"
#include "cudaErrors.cu"
#include <stdio.h>
#include <assert.h>

/****************************************************************************************/
/******************************* Set GPU propagation parameters *************************/
/****************************************************************************************/
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
void initBornGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc){

	// Set GPU number
	cudaSetDevice(iGpuId);

	host_nz = nz;
	host_nx = nx;
	host_dz = dz;
	host_dx = dx;
	host_nts = nts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;

	/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpuId == iGpuAlloc) {

		// Time slices for FD stepping
		dev_p0 = new double*[nGpu];
		dev_p1 = new double*[nGpu];
		dev_temp1 = new double*[nGpu];

		dev_ssLeft = new double*[nGpu];
		dev_ssRight = new double*[nGpu];
		dev_ssTemp1 = new double*[nGpu];

		// Data
		dev_dataRegDts = new double*[nGpu];

		// Source and receivers
		dev_sourcesPositionReg = new int*[nGpu];
		dev_receiversPositionReg = new int*[nGpu];

		// Sources signal
		dev_sourcesSignals = new double*[nGpu];

		// Scaled velocity
		dev_vel2Dtw2 = new double*[nGpu];

		// Reflectivity scaling
		dev_reflectivityScale = new double*[nGpu];

		// Reflectivity
		dev_modelBorn = new double*[nGpu];

		// Source wavefields
		dev_BornSrcWavefield = new double*[nGpu];

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
	int hInterpFilter = sub + 1;
	int nInterpFilter = 2 * hInterpFilter;

	// Check the subsampling coefficient is smaller than the maximum allowed
	if (sub>SUB_MAX){
		std::cout << "**** ERROR: Subsampling parameter is too high ****" << std::endl;
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
	// Check that the minimum padding is smaller than the max allowed
	if (minPad>PAD_MAX){
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

	// Check that the block size is consistent between parfile and "varDeclare.h"
	if (blockSize != BLOCK_SIZE) {
		std::cout << "**** ERROR: Block size for time stepper is not consistent with parfile ****" << std::endl;
		assert (1==2);
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
void allocateBornShotsGpu(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId){

	// Set GPU number
	cudaSetDevice(iGpuId);

	// Reflectivity scale
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nz*host_nx*sizeof(double))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice)); //

	// Scaled velocity
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nz*host_nx*sizeof(double))); // Allocate scaling for reflectivity
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice)); //

	// Allocate time slices
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nz*host_nx*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMalloc((void**) &dev_ssLeft[iGpu], host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMalloc((void**) &dev_ssRight[iGpu], host_nz*host_nx*sizeof(double)));

	// Allocate non-extended model
	cuda_call(cudaMalloc((void**) &dev_modelBorn[iGpu], host_nz*host_nx*sizeof(double)));

	// Allocate source wavefield
	cuda_call(cudaMalloc((void**) &dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(double))); // Allocate on device

}
void deallocateBornShotsGpu(int iGpu, int iGpuId){

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
		cuda_call(cudaFree(dev_modelBorn[iGpu]));
}

/****************************************************************************************/
/************************************** Born forward ************************************/
/****************************************************************************************/
void BornShotsFwdGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId){

	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

   	// Kernel parameters
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

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
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Copy model to device
	cuda_call(cudaMemcpy(dev_modelBorn[iGpu], model, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Apply both scalings to reflectivity: (1) 2.0*1/v^3 (2) v^2*dtw^2
	kernel_exec(scaleReflectivity<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Compute secondary source for first coarse time index (its = 0)
	kernel_exec(imagingFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_ssLeft[iGpu], 0, dev_BornSrcWavefield[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		kernel_exec(imagingFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_ssRight[iGpu], its+1, dev_BornSrcWavefield[iGpu]));

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGrid, dimBlock>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Extract data
			kernel_exec(recordInterpData<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}
void BornShotsFwdGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId){

	// Non-extended Born modeling operator (FORWARD)
	// The source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Allocate and initialize data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data at coarse time-sampling on device
  	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double))); // Initialize data on device

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

   	// Kernel parameters
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

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
	cuda_call(cudaMemcpy(srcWavefieldDts, dev_BornSrcWavefield[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/************************** Scattered wavefield computation *************************/
	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Allocate and copy model
	cuda_call(cudaMemcpy(dev_modelBorn[iGpu], model, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice)); // Copy model (reflectivity) on device

	// Allocate and initialize scattered wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(double))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

	// Apply both scalings to reflectivity:
	kernel_exec(scaleReflectivity<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Compute secondary source for first coarse time index (its = 0)
	kernel_exec(imagingFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_ssLeft[iGpu], 0, dev_BornSrcWavefield[iGpu]));

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		kernel_exec(imagingFwdGpu<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_ssRight[iGpu], its+1, dev_BornSrcWavefield[iGpu]));

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
			kernel_exec(recordInterpData<<<nblockData, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataRegDts[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Switch pointers for secondary source
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	}

	// Copy data back to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy scattered wavefield back to host
	cuda_call(cudaMemcpy(scatWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));

}

/****************************************************************************************/
/************************************** Born adjoint ************************************/
/****************************************************************************************/
void BornShotsAdjGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *recWavefieldDts, int iGpu, int iGpuId){

	// Non-extended Born modeling operator (ADJOINT)
	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

   	// Kernel parameters
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

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
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Model
  	cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nz*host_nx*sizeof(double))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy data on device

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

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

		// Apply imaging condition for its+1
  		kernel_exec(imagingAdjGpu<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Apply imaging condition for its=0
  	kernel_exec(imagingAdjGpu<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], 0));

  	// Scale model for finite-difference and secondary source coefficient
	kernel_exec(scaleReflectivity<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBorn[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));

}
void BornShotsAdjGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *recWavefieldDts, int iGpu, int iGpuId){

	// Non-extended Born modeling operator (ADJOINT)
	// We assume the source wavelet/signals already contain the second time derivative
	// Set device number
	cudaSetDevice(iGpuId);

	// Sources geometry
	cuda_call(cudaMemcpyToSymbol(dev_nSourcesReg, &nSourcesReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_sourcesPositionReg[iGpu], nSourcesReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_sourcesPositionReg[iGpu], sourcesPositionReg, nSourcesReg*sizeof(int), cudaMemcpyHostToDevice));

	// Sources signals
  	cuda_call(cudaMalloc((void**) &dev_sourcesSignals[iGpu], nSourcesReg*host_ntw*sizeof(double))); // Allocate sources signals on device
	cuda_call(cudaMemcpy(dev_sourcesSignals[iGpu], sourcesSignals, nSourcesReg*host_ntw*sizeof(double), cudaMemcpyHostToDevice)); // Copy sources signals on device

	// Receivers geometry
	cuda_call(cudaMemcpyToSymbol(dev_nReceiversReg, &nReceiversReg, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMalloc((void**) &dev_receiversPositionReg[iGpu], nReceiversReg*sizeof(int)));
	cuda_call(cudaMemcpy(dev_receiversPositionReg[iGpu], receiversPositionReg, nReceiversReg*sizeof(int), cudaMemcpyHostToDevice));

	// Initialize source wavefield on device
	cuda_call(cudaMemset(dev_BornSrcWavefield[iGpu], 0, host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

	// Initialize time-slices for time-stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

   	// Kernel parameters
	int nblockx = (host_nz-2*FAT) / BLOCK_SIZE;
	int nblocky = (host_nx-2*FAT) / BLOCK_SIZE;
	int nblockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA;
	dim3 dimGrid(nblockx, nblocky);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

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
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Allocate and initialize receiver wavefield on device
	cuda_call(cudaMalloc((void**) &dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(double))); // Allocate on device
	cuda_call(cudaMemset(dev_BornSecWavefield, 0, host_nz*host_nx*host_nts*sizeof(double))); // Initialize wavefield on device

	// Model
  	cuda_call(cudaMemset(dev_modelBorn[iGpu], 0, host_nz*host_nx*sizeof(double))); // Initialize model on device

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double))); // Allocate data on device
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice)); // Copy data on device

  	// Main loop
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step adjoint in time
			kernel_exec(stepAdjGpu<<<dimGrid, dimBlock>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nblockData, BLOCK_SIZE_DATA>>>(dev_dataRegDts[iGpu], dev_p0[iGpu], its, it2, dev_receiversPositionReg[iGpu]));

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

		// Apply imaging condition for its+1
  		kernel_exec(imagingAdjGpu<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], its+1));

		// Record and scale receiver wavefield at coarse sampling for its+1
		kernel_exec(recordScaleWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], its+1, dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

		// Switch pointers for receiver wavefield before imaging time derivative
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double))); // Reinitialize slice for coarse time-sampling before time derivative

	} // Finished main loop - we still have to compute imaging condition for its=0

	// Save receiver wavefield at its=0
	kernel_exec(recordScaleWavefield<<<dimGrid, dimBlock>>>(dev_BornSecWavefield, dev_ssRight[iGpu], 0, dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Apply imaging condition for its=0
  	kernel_exec(imagingAdjGpu<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_ssRight[iGpu], dev_BornSrcWavefield[iGpu], 0));

  	// Scale model for finite-difference and secondary source coefficient
	// It's better to apply it once and for all than at every time-steps
	kernel_exec(scaleReflectivity<<<dimGrid, dimBlock>>>(dev_modelBorn[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Copy model back to host
	cuda_call(cudaMemcpy(model, dev_modelBorn[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToHost));

	// Copy scattered wavefield back to host
	cuda_call(cudaMemcpy(recWavefieldDts, dev_BornSecWavefield, host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_BornSecWavefield));
}

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include "tomoExtShotsGpuFunctions.h"
#include "kernelsGpu.cu"
#include "cudaErrors.cu"
#include "varDeclare.h"
#include <ctime>
#include <stdio.h>
#include <assert.h>

/****************************************************************************************/
/************************ Declaration of auxiliary functions ****************************/
/****************************************************************************************/
// Note: The implementations of these auxiliary functions are done at the bottom of the file

void computeTomoSrcWfldDt2(double *dev_sourcesIn, double *dev_wavefieldOut, int *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu);
void computeTomoRecWfld(double *dev_dataIn, double *dev_wavefieldOut, int *dev_receiversPositionsRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu);

void computeTomoLeg1TimeFwd(double *dev_modelIn, double *dev_tomoSrcWavefieldDt2In, double *dev_dataOut, double *dev_wavefield1Out, double *dev_wavefield2Out, double *dev_extReflectivityIn, int *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, int saveWavefield);
void computeTomoLeg2TimeFwd(double *dev_modelIn, double *dev_tomoSrcWavefieldDt2In, double *dev_dataOut, double *dev_wavefield1Out, double *dev_wavefield2Out, double *dev_extReflectivityIn, int *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, int saveWavefield);

void computeTomoLeg1OffsetFwd(double *dev_modelIn, double *dev_tomoSrcWavefieldDt2In, double *dev_dataOut, double *dev_wavefield1Out, double *dev_wavefield2Out, double *dev_extReflectivityIn, int *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, int saveWavefield);
void computeTomoLeg2OffsetFwd(double *dev_modelIn, double *dev_tomoSrcWavefieldDt2In, double *dev_dataOut, double *dev_wavefield1Out, double *dev_wavefield2Out, double *dev_extReflectivityIn, int *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, int saveWavefield);

void computeTomoLeg1TimeAdj(double *dev_tomoSrcWavefieldDt2In, double *dev_tomoRecWavefieldIn, double *dev_modelTomoOut, double *dev_wavefield1Out, double *dev_extReflectivityIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, int saveWavefield);
void computeTomoLeg2TimeAdj(double *dev_tomoSrcWavefieldDt2In, double *dev_tomoRecWavefieldIn, double *dev_modelTomoOut, double *dev_wavefield1Out, double *dev_extReflectivityIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, int saveWavefield);

void computeTomoLeg1OffsetAdj(double *dev_tomoSrcWavefieldDt2In, double *dev_tomoRecWavefieldIn, double *dev_modelTomoOut, double *dev_wavefield1Out, double *dev_extReflectivityIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, int saveWavefield);
void computeTomoLeg2OffsetAdj(double *dev_tomoSrcWavefieldDt2In, double *dev_tomoRecWavefieldIn, double *dev_modelTomoOut, double *dev_wavefield1Out, double *dev_extReflectivityIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, int saveWavefield);

/****************************************************************************************/
/******************************* Set GPU propagation parameters *************************/
/****************************************************************************************/
// Display info on GPU
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
		cudaDriverGetVersion(&driver);
		std::cout << "Cuda driver version: " << driver << std::endl; // Driver

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

// Initialize GPU
void initTomoExtGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt, int leg1, int leg2, int nGpu, int iGpu){

	// Set GPU number
	cudaSetDevice(iGpu);

	host_nz = nz;
	host_nx = nx;
	host_dz = dz;
	host_dx = dx;
	host_nExt = nExt;
	host_hExt = (nExt-1)/2;
	host_nts = nts;
	host_dts = dts;
	host_sub = sub;
	host_ntw = (nts - 1) * sub + 1;
	host_leg1 = leg1;
	host_leg2 = leg2;

	// Coefficients for second-order time derivative
	host_cSide = 1.0 / (host_dts*host_dts);
	host_cCenter = -2.0 / (host_dts*host_dts);
	// host_cSide = 0.0;
	// host_cCenter = 1.0;

	/**************************** ALLOCATE ARRAYS OF ARRAYS *****************************/
	// Only one GPU will perform the following
	if (iGpu == 0) {

		// Time slices for FD stepping
		dev_p0 = new double*[nGpu];
		dev_p1 = new double*[nGpu];
		dev_temp1 = new double*[nGpu];

		// Time slices for FD stepping at coarse grid
		dev_ssLeft = new double*[nGpu];
		dev_ssRight = new double*[nGpu];
		dev_ssTemp1 = new double*[nGpu];

		// Time slices for FD stepping at coarse grid
		dev_scatLeft = new double*[nGpu];
		dev_scatRight = new double*[nGpu];
		dev_scatTemp1 = new double*[nGpu];

		// Time slices for FD stepping at coarse grid before second order time derivative
		dev_ss0 = new double*[nGpu];
		dev_ss1 = new double*[nGpu];
		dev_ss2 = new double*[nGpu];
		dev_ssTemp2 = new double*[nGpu];

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

		// Background perturbation ("model" for tomo)
		dev_modelTomo = new double*[nGpu];

		// Extended reflectivity for tomo
		dev_extReflectivity = new double*[nGpu];

		// Source and secondary wavefields
		dev_tomoSrcWavefieldDt2 = new double*[nGpu];
		dev_tomoSecWavefield1 = new double*[nGpu];
		dev_tomoSecWavefield2 = new double*[nGpu];

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
	int hInterpFilter = sub + 1;
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

	// Extension parameters
	cuda_call(cudaMemcpyToSymbol(dev_nExt, &host_nExt, sizeof(int), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_hExt, &host_hExt, sizeof(int), 0, cudaMemcpyHostToDevice));

	// Second order time derivative coefficients
	cuda_call(cudaMemcpyToSymbol(dev_cCenter, &host_cCenter, sizeof(double), 0, cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpyToSymbol(dev_cSide, &host_cSide, sizeof(double), 0, cudaMemcpyHostToDevice));

}

// Allocate on device
void allocateTomoExtShotsGpu(double *vel2Dtw2, double *reflectivityScale, double *extReflectivity, int iGpu){

	// Set GPU number
	cudaSetDevice(iGpu);

	// Velocity scale
	cuda_call(cudaMalloc((void**) &dev_vel2Dtw2[iGpu], host_nz*host_nx*sizeof(double))); // Allocate scaled velocity model on device
	cuda_call(cudaMemcpy(dev_vel2Dtw2[iGpu], vel2Dtw2, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
	cuda_call(cudaMemcpy(vel2Dtw2, dev_vel2Dtw2[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToHost));

	// Reflectivity scale
	cuda_call(cudaMalloc((void**) &dev_reflectivityScale[iGpu], host_nz*host_nx*sizeof(double))); // Allocate scaling for reflectivity
	cuda_call(cudaMemcpy(dev_reflectivityScale[iGpu], reflectivityScale, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));

	// Allocate time slices
	cuda_call(cudaMalloc((void**) &dev_p0[iGpu], host_nz*host_nx*sizeof(double))); // Fine time sampling temporary slices
	cuda_call(cudaMalloc((void**) &dev_p1[iGpu], host_nz*host_nx*sizeof(double)));

  	cuda_call(cudaMalloc((void**) &dev_ssLeft[iGpu], host_nz*host_nx*sizeof(double))); // Coarse time sampling temporary slices
  	cuda_call(cudaMalloc((void**) &dev_ssRight[iGpu], host_nz*host_nx*sizeof(double)));

	cuda_call(cudaMalloc((void**) &dev_scatLeft[iGpu], host_nz*host_nx*sizeof(double))); // Coarse time sampling temporary slices
  	cuda_call(cudaMalloc((void**) &dev_scatRight[iGpu], host_nz*host_nx*sizeof(double)));

	cuda_call(cudaMalloc((void**) &dev_ss0[iGpu], host_nz*host_nx*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_ss1[iGpu], host_nz*host_nx*sizeof(double)));
	cuda_call(cudaMalloc((void**) &dev_ss2[iGpu], host_nz*host_nx*sizeof(double)));

	// Allocate non-extended model
	cuda_call(cudaMalloc((void**) &dev_modelTomo[iGpu], host_nz*host_nx*sizeof(double)));

	// Allocate and copy from host to device extended reflectivity
	cuda_call(cudaMalloc((void**) &dev_extReflectivity[iGpu], host_nz*host_nx*host_nExt*sizeof(double)));
	cuda_call(cudaMemcpy(dev_extReflectivity[iGpu], extReflectivity, host_nz*host_nx*host_nExt*sizeof(double), cudaMemcpyHostToDevice));

	// Scale reflectivity
	int nBlockZExt = (host_nz-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the z-axis when using a time-extension
	int nBlockXExt = (host_nx-2*FAT) / BLOCK_SIZE_EXT; // Number of blocks for the x-axis when using a time-extension
	int nBlockExt = (host_nExt+BLOCK_SIZE_EXT-1) / BLOCK_SIZE_EXT;
	dim3 dimGridExt(nBlockZExt, nBlockXExt, nBlockExt);
	dim3 dimBlockExt(BLOCK_SIZE_EXT, BLOCK_SIZE_EXT, BLOCK_SIZE_EXT);
	kernel_exec(scaleReflectivityExt<<<dimGridExt, dimBlockExt>>>(dev_extReflectivity[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Allocate source wavefield - The allocation of secondary sources are done inside the function implementation
	cuda_call(cudaMalloc((void**) &dev_tomoSrcWavefieldDt2[iGpu], host_nz*host_nx*host_nts*sizeof(double))); // We store the source wavefield

	// Allocate the auxiliary wavefield (either scattered for the fwd, or receiver for the adjoint)
	cuda_call(cudaMalloc((void**) &dev_tomoSecWavefield1[iGpu], host_nz*host_nx*host_nts*sizeof(double))); // We store the source wavefield
}

// Deallocate from device
void deallocateTomoExtShotsGpu(int iGpu){

 		// Set device number on GPU cluster
		cudaSetDevice(iGpu);

		// Deallocate all DAT shit
    	cuda_call(cudaFree(dev_vel2Dtw2[iGpu]));
    	cuda_call(cudaFree(dev_reflectivityScale[iGpu]));
		cuda_call(cudaFree(dev_p0[iGpu]));
    	cuda_call(cudaFree(dev_p1[iGpu]));
		cuda_call(cudaFree(dev_ssLeft[iGpu]));
		cuda_call(cudaFree(dev_ssRight[iGpu]));
		cuda_call(cudaFree(dev_ss0[iGpu]));
		cuda_call(cudaFree(dev_ss1[iGpu]));
		cuda_call(cudaFree(dev_ss2[iGpu]));
		cuda_call(cudaFree(dev_tomoSrcWavefieldDt2[iGpu]));
		cuda_call(cudaFree(dev_tomoSecWavefield1[iGpu]));
		cuda_call(cudaFree(dev_extReflectivity[iGpu]));
		cuda_call(cudaFree(dev_modelTomo[iGpu]));
}

/****************************************************************************************/
/************************************** Tomo forward ************************************/
/****************************************************************************************/
void tomoExtShotsFwdGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int saveWavefield, std::string extension){

	// We assume the source wavelet/signals already contain(s) the second time derivative
	// Set device number
	cudaSetDevice(iGpu);

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

	// Non-extended blocks/threads
	int nBlockZ = (host_nz-2*FAT) / BLOCK_SIZE; // Number of blocks for the z-axis
	int nBlockX = (host_nx-2*FAT) / BLOCK_SIZE; // Number of blocks for the x-axis
	int nBlockData = (nReceiversReg+BLOCK_SIZE_DATA-1) / BLOCK_SIZE_DATA; // Number of blocks for the data extraction/injection
	dim3 dimGrid(nBlockZ, nBlockX);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	/************************************************************************************/
	/*************************************** Source *************************************/
	/************************************************************************************/
	// Compute source wavefield with second-order time derivative
	computeTomoSrcWfldDt2(dev_sourcesSignals[iGpu], dev_tomoSrcWavefieldDt2[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, iGpu);

	// Copy source wavefield back to host
	if (saveWavefield == 1) {cuda_call(cudaMemcpy(tomoSrcWavefieldDt2, dev_tomoSrcWavefieldDt2[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));}

	/************************************************************************************/
	/***************************** Preliminary steps ************************************/
	/************************************************************************************/
	// Copy + scale model (background perturbation)
	cuda_call(cudaMemcpy(dev_modelTomo[iGpu], model, host_nz*host_nx*sizeof(double), cudaMemcpyHostToDevice));
	kernel_exec(scaleReflectivity<<<dimGrid, dimBlock>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Allocate data and initialize to zero
	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double)));
	cuda_call(cudaMemset(dev_dataRegDts[iGpu], 0, nReceiversReg*host_nts*sizeof(double)));

	// Allocate secondary wavefields
	if (saveWavefield == 1) {cuda_call(cudaMalloc((void**) &dev_tomoSecWavefield2[iGpu], host_nz*host_nx*host_nts*sizeof(double)));}

	/************************************************************************************/
	/************************************ Leg #1 ****************************************/
	/************************************************************************************/
	if (host_leg1 == 1){

		// source->reflectivity->model->data
		if (extension == "time") {

			computeTomoLeg1TimeFwd(dev_modelTomo[iGpu], dev_tomoSrcWavefieldDt2[iGpu], dev_dataRegDts[iGpu], dev_tomoSecWavefield1[iGpu], dev_tomoSecWavefield2[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, nBlockData, iGpu, saveWavefield);

		} else {

			computeTomoLeg1OffsetFwd(dev_modelTomo[iGpu], dev_tomoSrcWavefieldDt2[iGpu], dev_dataRegDts[iGpu], dev_tomoSecWavefield1[iGpu], dev_tomoSecWavefield2[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, nBlockData, iGpu, saveWavefield);

		}

		// Copy both scattered wavefields from leg #1 to host
		if (saveWavefield == 1) {
			cuda_call(cudaMemcpy(tomoSecWavefield1, dev_tomoSecWavefield1[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
			cuda_call(cudaMemcpy(tomoSecWavefield2, dev_tomoSecWavefield2[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		}
	}

	/************************************************************************************/
	/************************************ Leg #2 ****************************************/
	/************************************************************************************/
	if (host_leg2 == 1){

		// source->model->reflectivity->data
		if (extension == "time"){

			computeTomoLeg2TimeFwd(dev_modelTomo[iGpu], dev_tomoSrcWavefieldDt2[iGpu], dev_dataRegDts[iGpu], dev_tomoSecWavefield1[iGpu], dev_tomoSecWavefield2[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, nBlockData, iGpu, saveWavefield);

		} else {

			computeTomoLeg2OffsetFwd(dev_modelTomo[iGpu], dev_tomoSrcWavefieldDt2[iGpu], dev_dataRegDts[iGpu], dev_tomoSecWavefield1[iGpu], dev_tomoSecWavefield2[iGpu], dev_extReflectivity[iGpu], dev_receiversPositionReg[iGpu], nReceiversReg, dimGrid, dimBlock, nBlockData, iGpu, saveWavefield);

		}

		// Copy scattered wavefields from leg #2 to host
		if (saveWavefield == 1) {
			cuda_call(cudaMemcpy(tomoSecWavefield1, dev_tomoSecWavefield1[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
			cuda_call(cudaMemcpy(tomoSecWavefield2, dev_tomoSecWavefield2[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));
		}
	}

	/************************************************************************************/
	/************************************ Data ******************************************/
	/************************************************************************************/
	// Copy data to host
	cuda_call(cudaMemcpy(dataRegDts, dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate all slices
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
	if (saveWavefield == 1){ cuda_call(cudaFree(dev_tomoSecWavefield2[iGpu]));}

}

/****************************************************************************************/
/************************************** Tomo adjoint ************************************/
/****************************************************************************************/
void tomoExtShotsAdjGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int saveWavefield, std::string extension){

	// We assume the source wavelet/signals already contain(s) the second time derivative
	// Set device number
	cudaSetDevice(iGpu);

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

	// Data
  	cuda_call(cudaMalloc((void**) &dev_dataRegDts[iGpu], nReceiversReg*host_nts*sizeof(double)));
	cuda_call(cudaMemcpy(dev_dataRegDts[iGpu], dataRegDts, nReceiversReg*host_nts*sizeof(double), cudaMemcpyHostToDevice));

	// Initialize models
	cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nz*host_nx*sizeof(double)));

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

	/************************************************************************************/
	/*************************************** Source *************************************/
	/************************************************************************************/
	// Compute source wavefield with second-order time derivative
	computeTomoSrcWfldDt2(dev_sourcesSignals[iGpu], dev_tomoSrcWavefieldDt2[iGpu], dev_sourcesPositionReg[iGpu], nSourcesReg, dimGrid, dimBlock, iGpu);
	if (saveWavefield == 1) {cuda_call(cudaMemcpy(tomoSrcWavefieldDt2, dev_tomoSrcWavefieldDt2[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));}

	/************************************************************************************/
	/********************************** Receiver ****************************************/
	/************************************************************************************/
	// Compute receiver wavefield (include no time derivative)
	computeTomoRecWfld(dev_dataRegDts[iGpu], dev_tomoSecWavefield1[iGpu], dev_receiversPositionReg[iGpu], dimGrid, dimBlock, nBlockData, iGpu);
	if (saveWavefield == 1) {cuda_call(cudaMemcpy(tomoSecWavefield1, dev_tomoSecWavefield1[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));}

	/************************************************************************************/
	/***************************** Preliminary steps ************************************/
	/************************************************************************************/
	// Allocate second secondary wavefield
	if (saveWavefield == 1) {cuda_call(cudaMalloc((void**) &dev_tomoSecWavefield2[iGpu], host_nz*host_nx*host_nts*sizeof(double)));}

	// Initialize model for both legs
    cuda_call(cudaMemset(dev_modelTomo[iGpu], 0, host_nz*host_nx*sizeof(double)));

	/************************************************************************************/
	/************************************ Leg #1 ****************************************/
	/************************************************************************************/
	if (host_leg1 == 1){

		if (extension == "time"){
			computeTomoLeg1TimeAdj(dev_tomoSrcWavefieldDt2[iGpu], dev_tomoSecWavefield1[iGpu], dev_modelTomo[iGpu], dev_tomoSecWavefield2[iGpu], dev_extReflectivity[iGpu], dimGrid, dimBlock, iGpu, saveWavefield);
		} else {
			computeTomoLeg1OffsetAdj(dev_tomoSrcWavefieldDt2[iGpu], dev_tomoSecWavefield1[iGpu], dev_modelTomo[iGpu], dev_tomoSecWavefield2[iGpu], dev_extReflectivity[iGpu], dimGrid, dimBlock, iGpu, saveWavefield);
		}

		// Copy scattered wavefield from leg #1 to host
		if (saveWavefield == 1) {cuda_call(cudaMemcpy(tomoSecWavefield2, dev_tomoSecWavefield2[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));}

	}

	/************************************************************************************/
	/************************************ Leg #2 ****************************************/
	/************************************************************************************/
	if (host_leg2 == 1){

		if (extension == "time"){
			computeTomoLeg2TimeAdj(dev_tomoSrcWavefieldDt2[iGpu], dev_tomoSecWavefield1[iGpu], dev_modelTomo[iGpu], dev_tomoSecWavefield2[iGpu], dev_extReflectivity[iGpu], dimGrid, dimBlock, iGpu, saveWavefield);
		} else {
			computeTomoLeg2OffsetAdj(dev_tomoSrcWavefieldDt2[iGpu], dev_tomoSecWavefield1[iGpu], dev_modelTomo[iGpu], dev_tomoSecWavefield2[iGpu], dev_extReflectivity[iGpu], dimGrid, dimBlock, iGpu, saveWavefield);
		}

		// Copy scattered wavefield from leg #2 to host
		if (saveWavefield == 1) {cuda_call(cudaMemcpy(tomoSecWavefield2, dev_tomoSecWavefield2[iGpu], host_nz*host_nx*host_nts*sizeof(double), cudaMemcpyDeviceToHost));}

	}

	/************************************************************************************/
	/************************************ Model *****************************************/
	/************************************************************************************/
	// Scale model
	kernel_exec(scaleReflectivity<<<dimGrid, dimBlock>>>(dev_modelTomo[iGpu], dev_reflectivityScale[iGpu], dev_vel2Dtw2[iGpu]));

	// Copy data to host
	cuda_call(cudaMemcpy(model, dev_modelTomo[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToHost));

	/******************************* Deallocation ***************************************/
	// Deallocate model, data and wavefields
    cuda_call(cudaFree(dev_sourcesPositionReg[iGpu]));
    cuda_call(cudaFree(dev_sourcesSignals[iGpu]));
    cuda_call(cudaFree(dev_dataRegDts[iGpu]));
    cuda_call(cudaFree(dev_receiversPositionReg[iGpu]));
	if (saveWavefield == 1) {cuda_call(cudaFree(dev_tomoSecWavefield2[iGpu]));}
}

/****************************************************************************************/
/********************************** Auxiliary functions *********************************/
/****************************************************************************************/

/************************************* Common parts *************************************/

// Source wavefield
void computeTomoSrcWfldDt2(double *dev_sourcesIn, double *dev_wavefieldOut, int *dev_sourcesPositionsRegIn, int nSourcesRegIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu){

	// Initialize wavefield on device
	cuda_call(cudaMemset(dev_wavefieldOut, 0, host_nz*host_nx*host_nts*sizeof(double)));

	// Initialize time-slices for time stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
   	cuda_call(cudaMemset(dev_ss0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ss1[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ss2[iGpu], 0, host_nz*host_nx*sizeof(double)));

    // Compute coarse source wavefield sample for its=0
	int its = 0;
	for (int it2 = 1; it2 < host_sub+1; it2++){

		// Compute fine time-step index
		int itw = its * host_sub + it2;

		// Step forward
		kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

		// Inject source
		kernel_exec(injectSource<<<1, nSourcesRegIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionsRegIn));

		// Damp wavefields
		kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));

		// Spread energy into dev_ss1 and dev_ss2
		kernel_exec(interpFineToCoarseSlice<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2));

		// Switch pointers
		dev_temp1[iGpu] = dev_p0[iGpu];
		dev_p0[iGpu] = dev_p1[iGpu];
		dev_p1[iGpu] = dev_temp1[iGpu];
		dev_temp1[iGpu] = NULL;

	}

	// Copy ss1 (its=0)
	cuda_call(cudaMemcpy(dev_ss1[iGpu], dev_ssLeft[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));

	// Switch coarse grid pointers
	dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
	dev_ssLeft[iGpu] = dev_ssRight[iGpu];
	dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
	dev_ssTemp1[iGpu] = NULL;

	for (int its=1; its<host_nts-1; its++){

	    for (int it2=1; it2<host_sub+1; it2++){

	        // Compute fine time-step index
	        int itw = its * host_sub + it2;

	        // Step forward
	        kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

	        // Inject source
	        kernel_exec(injectSource<<<1, nSourcesRegIn>>>(dev_sourcesIn, dev_p0[iGpu], itw-1, dev_sourcesPositionsRegIn));

	        // Damp wavefields
	        kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));

	        // Spread energy into dev_ssLeft and dev_ssRight
	        kernel_exec(interpFineToCoarseSlice<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2));

	        // Switch pointers
	        dev_temp1[iGpu] = dev_p0[iGpu];
	        dev_p0[iGpu] = dev_p1[iGpu];
	        dev_p1[iGpu] = dev_temp1[iGpu];
	        dev_temp1[iGpu] = NULL;

	    }

		// Copy ss2 (value of source wavefield at its
	    cuda_call(cudaMemcpy(dev_ss2[iGpu], dev_ssLeft[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));

	    // Compute second order time derivative of source wavefield at its-1
	    kernel_exec(srcWfldSecondTimeDerivative<<<dimGridIn, dimBlockIn>>>(dev_wavefieldOut, dev_ss0[iGpu], dev_ss1[iGpu], dev_ss2[iGpu], its-1));

	    // Switch coarse time sampling pointers
	    dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
	    dev_ssLeft[iGpu] = dev_ssRight[iGpu];
	    dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
	    cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
	    dev_ssTemp1[iGpu] = NULL;

	    // Switch pointers for time derivative
	    dev_ssTemp2[iGpu] = dev_ss0[iGpu];
	    dev_ss0[iGpu] = dev_ss1[iGpu];
	    dev_ss1[iGpu] = dev_ss2[iGpu];
	    dev_ss2[iGpu] = dev_ssTemp2[iGpu];
	    dev_ssTemp2[iGpu] = NULL;
	}

	// Copy ssLeft to ss2 which corresponds to wavefield value (before time derivative) at nts-1
	cuda_call(cudaMemcpy(dev_ss2[iGpu], dev_ssLeft[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));

	// Compute second order time derivative at nts-2
	kernel_exec(srcWfldSecondTimeDerivative<<<dimGridIn, dimBlockIn>>>(dev_wavefieldOut, dev_ss0[iGpu], dev_ss1[iGpu], dev_ss2[iGpu], host_nts-2));

	// Compute second order time derivative at nts-1 (now ss2 is in the middle of the stencil)
	cuda_call(cudaMemset(dev_ss0[iGpu], 0, host_nz*host_nx*sizeof(double)));
	kernel_exec(srcWfldSecondTimeDerivative<<<dimGridIn, dimBlockIn>>>(dev_wavefieldOut, dev_ss0[iGpu], dev_ss2[iGpu], dev_ss1[iGpu], host_nts-1));

}

// Receiver wavefield
void computeTomoRecWfld(double *dev_dataIn, double *dev_wavefieldOut, int *dev_receiversPositionsRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu){

	// Initialize wavefield on device
	cuda_call(cudaMemset(dev_wavefieldOut, 0, host_nz*host_nx*host_nts*sizeof(double)));

	// Initialize time-slices for time stepping
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Start propagation
	for (int its = host_nts-2; its > -1; its--){

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step forward in time
			kernel_exec(stepAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject data
			kernel_exec(interpInjectData<<<nBlockDataIn, BLOCK_SIZE_DATA>>>(dev_dataIn, dev_p0[iGpu], its, it2, dev_receiversPositionsRegIn));

			// Damp wavefield
			kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate and save wavefield on device (the wavefield is not scaled)
			kernel_exec(interpWavefield<<<dimGridIn, dimBlockIn>>>(dev_wavefieldOut, dev_p0[iGpu], its, it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;
		}
	}
}

/*************************************** Forward ****************************************/

// Leg 1 forward [time]: s -> i -> m -> d
void computeTomoLeg1TimeFwd(double *dev_modelIn, double *dev_tomoSrcWavefieldDt2In, double *dev_dataOut, double *dev_wavefield1Out, double *dev_wavefield2Out, double *dev_extReflectivityIn, int *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, int saveWavefield){

	/**************** First part of leg #1 ****************/

	// Initialize scattered wavefield #1
    cuda_call(cudaMemset(dev_wavefield1Out, 0, host_nz*host_nx*host_nts*sizeof(double))); // Note: We need to initialize the wavefield because the kernel interpWavefiel uses +=

    // Initialize slices
    cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Compute secondary source from extended scattering condition for first coarse time index (its = 0)
    int its = 0;
	int iExtMin, iExtMax;
    iExtMin = (its+1-host_nts)/2;
    iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
    iExtMax = its/2;
    iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Add 1 for the strict inequality in the "for loop"
    kernel_exec(imagingTimeFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssLeft[iGpu], dev_tomoSrcWavefieldDt2In, its, iExtMin, iExtMax)); // Apply extended FWD imaging condition

    // Start propagating scattered wavefield
    for (int its = 0; its < host_nts-1; its++){

        // Compute secondary source for first coarse time index (its+1)
        iExtMin = (its+2-host_nts)/2;
        iExtMin = std::max(iExtMin, -host_hExt) + host_hExt; // Lower bound for extended index
        iExtMax = (its+1)/2;
        iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for extended index
        kernel_exec(imagingTimeFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssRight[iGpu], dev_tomoSrcWavefieldDt2In, its+1, iExtMin, iExtMax)); // Apply time-extended FWD imaging condition

        for (int it2 = 1; it2 < host_sub+1; it2++){

            // Step forward
            kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
            kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));
            kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));
            kernel_exec(interpWavefield<<<dimGridIn, dimBlockIn>>>(dev_wavefield1Out, dev_p0[iGpu], its, it2));

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
        dev_ssTemp1[iGpu] = NULL;

        // Reinitialize slice for right coarse sample. We have to do it for the extended imaging condition because we update dev_ssRight with += in the kernel
        // We don't have to do it for the non-extended imaging condition because there is no +=
        cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

    }

	/**************** Second part of leg #1 ****************/
	if (saveWavefield == 1) {cuda_call(cudaMemset(dev_wavefield2Out, 0, host_nz*host_nx*host_nts*sizeof(double)));} // Note: We need to initialize the wavefield because the kernel interpWavefiel uses +=

	// Initialize slices
    cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Compute secondary source for first coarse time index (its=0)
    kernel_exec(imagingFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_modelIn, dev_ssLeft[iGpu], 0, dev_wavefield1Out));

    // Start propagating scattered wavefield
    for (int its = 0; its < host_nts-1; its++){

        // Compute secondary source for first coarse time index (its+1)
        kernel_exec(imagingFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_modelIn, dev_ssRight[iGpu], its+1, dev_wavefield1Out));

        for (int it2 = 1; it2 < host_sub+1; it2++){

            // Update wavefield
            kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
            kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));
            kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));
            if (saveWavefield == 1) {kernel_exec(interpWavefield<<<dimGridIn, dimBlockIn>>>(dev_wavefield2Out, dev_p0[iGpu], its, it2));}
            kernel_exec(recordInterpData<<<nBlockDataIn, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataOut, its, it2, dev_receiversPositionRegIn));

            // Switch pointers L1
            dev_temp1[iGpu] = dev_p0[iGpu];
            dev_p0[iGpu] = dev_p1[iGpu];
            dev_p1[iGpu] = dev_temp1[iGpu];
            dev_temp1[iGpu] = NULL;

        }

        // Switch pointers for secondary source
        dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
        dev_ssLeft[iGpu] = dev_ssRight[iGpu];
        dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
        dev_ssTemp1[iGpu] = NULL;
    }
}

// Leg 2 forward [time]: s -> m -> i -> d
void computeTomoLeg2TimeFwd(double *dev_modelIn, double *dev_tomoSrcWavefieldDt2In, double *dev_dataOut, double *dev_wavefield1Out, double *dev_wavefield2Out, double *dev_extReflectivityIn, int *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, int saveWavefield){

	// Initialize scattered wavefield #1
    cuda_call(cudaMemset(dev_wavefield1Out, 0, host_nz*host_nx*host_nts*sizeof(double)));

    // Initialize slices
    cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	/************************** Scattered wavefield #1 ************************/
    // Compute secondary source for first coarse time index (its=0)
    kernel_exec(imagingFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_modelIn, dev_ssLeft[iGpu], 0, dev_tomoSrcWavefieldDt2In));

    // Start propagating scattered wavefield
    for (int its = 0; its < host_nts-1; its++){

        // Compute secondary source for first coarse time index (its+1)
        kernel_exec(imagingFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_modelIn, dev_ssRight[iGpu], its+1, dev_tomoSrcWavefieldDt2In));

        for (int it2 = 1; it2 < host_sub+1; it2++){

            // Update wavefield
            kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
            kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));
            kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));
            kernel_exec(interpWavefield<<<dimGridIn, dimBlockIn>>>(dev_wavefield1Out, dev_p0[iGpu], its, it2));

            // Switch pointers L1
            dev_temp1[iGpu] = dev_p0[iGpu];
            dev_p0[iGpu] = dev_p1[iGpu];
            dev_p1[iGpu] = dev_temp1[iGpu];
            dev_temp1[iGpu] = NULL;

        }

        // Switch pointers for secondary source
        dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
        dev_ssLeft[iGpu] = dev_ssRight[iGpu];
        dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
        dev_ssTemp1[iGpu] = NULL;
    }

	/************************** Scattered wavefield #2 ************************/
    // Initialize scattered wavefield
    if (saveWavefield == 1) {cuda_call(cudaMemset(dev_wavefield2Out, 0, host_nz*host_nx*host_nts*sizeof(double)));}

    // Initialize slices
    cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

    // Compute secondary source from extended scattering condition for first coarse time index (its=0)
    int its = 0;
	int iExtMin, iExtMax;
    iExtMin = (its+1-host_nts)/2;
    iExtMin = std::max(iExtMin, -host_hExt) + host_hExt; // Modified
    iExtMax = its/2;
    iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Add 1 for the strict inequality in the "for loop"
    kernel_exec(imagingTimeFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssLeft[iGpu], dev_wavefield1Out, its, iExtMin, iExtMax)); // Apply extended FWD imaging condition

    // Start propagating scattered wavefield
    for (int its = 0; its < host_nts-1; its++){

        // Compute secondary source for first coarse time index (its+1)
        iExtMin = (its+2-host_nts)/2;
        iExtMin = std::max(iExtMin, -host_hExt) + host_hExt; // Lower bound for extended index
        iExtMax = (its+1)/2;
        iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for extended index
        kernel_exec(imagingTimeFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssRight[iGpu], dev_wavefield1Out, its+1, iExtMin, iExtMax)); // Apply time-extended FWD imaging condition

        for (int it2 = 1; it2 < host_sub+1; it2++){

            // Step forward
            kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
            kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));
            kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));
            if (saveWavefield == 1) {kernel_exec(interpWavefield<<<dimGridIn, dimBlockIn>>>(dev_wavefield2Out, dev_p0[iGpu], its, it2));}
            kernel_exec(recordInterpData<<<nBlockDataIn, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataOut, its, it2, dev_receiversPositionRegIn));

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
        dev_ssTemp1[iGpu] = NULL;
        cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double))); // Reinitialize slice for coarse
    }

}

// Leg 1 forward [offset]: s -> i -> m -> d
void computeTomoLeg1OffsetFwd(double *dev_modelIn, double *dev_tomoSrcWavefieldDt2In, double *dev_dataOut, double *dev_wavefield1Out, double *dev_wavefield2Out, double *dev_extReflectivityIn, int *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, int saveWavefield){

	/**************** First part of leg #1 ****************/

	// Initialize scattered wavefield #1
    cuda_call(cudaMemset(dev_wavefield1Out, 0, host_nz*host_nx*host_nts*sizeof(double))); // Note: We need to initialize the wavefield because the kernel interpWavefiel uses +=

    // Initialize slices
    cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Compute secondary source from extended scattering condition for first coarse time index (its = 0)
    int its = 0;
	kernel_exec(imagingOffsetFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssLeft[iGpu], dev_tomoSrcWavefieldDt2In, its)); // Apply extended FWD imaging condition

    // Start propagating scattered wavefield
    for (int its = 0; its < host_nts-1; its++){

        // Compute secondary source for first coarse time index (its+1)
        kernel_exec(imagingOffsetFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssRight[iGpu], dev_tomoSrcWavefieldDt2In, its+1)); // Apply time-extended FWD imaging condition

        for (int it2 = 1; it2 < host_sub+1; it2++){

            // Step forward
            kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
            kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));
            kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));
            kernel_exec(interpWavefield<<<dimGridIn, dimBlockIn>>>(dev_wavefield1Out, dev_p0[iGpu], its, it2));

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
        dev_ssTemp1[iGpu] = NULL;

        // Reinitialize slice for right coarse sample. We have to do it for the extended imaging condition because we update dev_ssRight with += in the kernel
        // We don't have to do it for the non-extended imaging condition because there is no +=
        cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

    }

	/**************** Second part of leg #1 ****************/
	if (saveWavefield == 1) {cuda_call(cudaMemset(dev_wavefield2Out, 0, host_nz*host_nx*host_nts*sizeof(double)));} // Note: We need to initialize the wavefield because the kernel interpWavefiel uses +=

	// Initialize slices
    cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Compute secondary source for first coarse time index (its=0)
    kernel_exec(imagingFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_modelIn, dev_ssLeft[iGpu], 0, dev_wavefield1Out));

    // Start propagating scattered wavefield
    for (int its = 0; its < host_nts-1; its++){

        // Compute secondary source for first coarse time index (its+1)
        kernel_exec(imagingFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_modelIn, dev_ssRight[iGpu], its+1, dev_wavefield1Out));

        for (int it2 = 1; it2 < host_sub+1; it2++){

            // Update wavefield
            kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
            kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));
            kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));
            if (saveWavefield == 1) {kernel_exec(interpWavefield<<<dimGridIn, dimBlockIn>>>(dev_wavefield2Out, dev_p0[iGpu], its, it2));}
            kernel_exec(recordInterpData<<<nBlockDataIn, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataOut, its, it2, dev_receiversPositionRegIn));

            // Switch pointers L1
            dev_temp1[iGpu] = dev_p0[iGpu];
            dev_p0[iGpu] = dev_p1[iGpu];
            dev_p1[iGpu] = dev_temp1[iGpu];
            dev_temp1[iGpu] = NULL;

        }

        // Switch pointers for secondary source
        dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
        dev_ssLeft[iGpu] = dev_ssRight[iGpu];
        dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
        dev_ssTemp1[iGpu] = NULL;
    }
}

// Leg 2 forward [offset]: s -> m -> i -> d
void computeTomoLeg2OffsetFwd(double *dev_modelIn, double *dev_tomoSrcWavefieldDt2In, double *dev_dataOut, double *dev_wavefield1Out, double *dev_wavefield2Out, double *dev_extReflectivityIn, int *dev_receiversPositionRegIn, int nReceiversRegIn, dim3 dimGridIn, dim3 dimBlockIn, int nBlockDataIn, int iGpu, int saveWavefield){

	// Initialize scattered wavefield #1
    cuda_call(cudaMemset(dev_wavefield1Out, 0, host_nz*host_nx*host_nts*sizeof(double)));

    // Initialize slices
    cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	/************************** Scattered wavefield #1 ************************/
    // Compute secondary source for first coarse time index (its=0)
    kernel_exec(imagingFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_modelIn, dev_ssLeft[iGpu], 0, dev_tomoSrcWavefieldDt2In));

    // Start propagating scattered wavefield
    for (int its = 0; its < host_nts-1; its++){

        // Compute secondary source for first coarse time index (its+1)
        kernel_exec(imagingFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_modelIn, dev_ssRight[iGpu], its+1, dev_tomoSrcWavefieldDt2In));

        for (int it2 = 1; it2 < host_sub+1; it2++){

            // Update wavefield
            kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
            kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));
            kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));
            kernel_exec(interpWavefield<<<dimGridIn, dimBlockIn>>>(dev_wavefield1Out, dev_p0[iGpu], its, it2));

            dev_temp1[iGpu] = dev_p0[iGpu];
            dev_p0[iGpu] = dev_p1[iGpu];
            dev_p1[iGpu] = dev_temp1[iGpu];
            dev_temp1[iGpu] = NULL;

        }

        // Switch pointers for secondary source
        dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
        dev_ssLeft[iGpu] = dev_ssRight[iGpu];
        dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
        dev_ssTemp1[iGpu] = NULL;
    }

	/************************** Scattered wavefield #2 ************************/
    // Initialize scattered wavefield
    if (saveWavefield == 1) {cuda_call(cudaMemset(dev_wavefield2Out, 0, host_nz*host_nx*host_nts*sizeof(double)));}

    // Initialize slices
    cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
    cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

    // Compute secondary source from extended scattering condition for first coarse time index (its=0)
    int its = 0;
    kernel_exec(imagingOffsetFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssLeft[iGpu], dev_wavefield1Out, its)); // Apply extended FWD imaging condition

    // Start propagating scattered wavefield
    for (int its = 0; its < host_nts-1; its++){

        // Compute secondary source for first coarse time index (its+1)
        kernel_exec(imagingOffsetFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssRight[iGpu], dev_wavefield1Out, its+1)); // Apply time-extended FWD imaging condition

        for (int it2 = 1; it2 < host_sub+1; it2++){

            // Step forward
            kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));
            kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));
            kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));
            if (saveWavefield == 1) {kernel_exec(interpWavefield<<<dimGridIn, dimBlockIn>>>(dev_wavefield2Out, dev_p0[iGpu], its, it2));}
            kernel_exec(recordInterpData<<<nBlockDataIn, BLOCK_SIZE_DATA>>>(dev_p0[iGpu], dev_dataOut, its, it2, dev_receiversPositionRegIn));

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
        dev_ssTemp1[iGpu] = NULL;
        cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double))); // Reinitialize slice for coarse
    }

}

// Leg 1 adjoint [time]: s -> i -> m <- d
void computeTomoLeg1TimeAdj(double *dev_tomoSrcWavefieldDt2In, double *dev_tomoRecWavefieldIn, double *dev_modelTomoOut, double *dev_wavefield1Out, double *dev_extReflectivityIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, int saveWavefield){

	// Scatter source off of extended reflectivity and cross-correlate with receiver wavefield

	// Initialize scattered wavefield
	if (saveWavefield == 1){ cuda_call(cudaMemset(dev_wavefield1Out, 0, host_nz*host_nx*host_nts*sizeof(double))); }

	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
	cuda_call(cudaMemset(dev_scatLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_scatRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Compute secondary source for first coarse time index (its = 0)
	int its = 0;
	int iExtMin, iExtMax;
	iExtMin = (its+1-host_nts)/2;
	iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	iExtMax = its/2;
	iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Add 1 for the strict inequality in the "for loop"
	kernel_exec(imagingTimeFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssLeft[iGpu], dev_tomoSrcWavefieldDt2In, its, iExtMin, iExtMax)); // Apply extended FWD imaging condition

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		iExtMin = (its+2-host_nts)/2;
		iExtMin = std::max(iExtMin, -host_hExt) + host_hExt; // Lower bound for extended index
		iExtMax = (its+1)/2;
		iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Upper bound for extended index
		kernel_exec(imagingTimeFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssRight[iGpu], dev_tomoSrcWavefieldDt2In, its+1, iExtMin, iExtMax)); // Apply time-extended FWD imaging condition

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate fine time slice to coarse time slice
			kernel_exec(interpFineToCoarseSlice<<<dimGridIn, dimBlockIn>>>(dev_scatLeft[iGpu], dev_scatRight[iGpu], dev_p0[iGpu], it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its
		kernel_exec(imagingAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_modelTomoOut, dev_scatLeft[iGpu], dev_tomoRecWavefieldIn, its));

		// Copy slice at its to scattered wavefield
		if (saveWavefield == 1) {cuda_call(cudaMemcpy(dev_wavefield1Out+its*host_nz*host_nx, dev_scatLeft[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));}

		// Switch pointers for secondary source
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

		// Switch pointers scattered wavefield
		dev_scatTemp1[iGpu] = dev_scatLeft[iGpu];
		dev_scatLeft[iGpu] = dev_scatRight[iGpu];
		dev_scatRight[iGpu] = dev_scatTemp1[iGpu];
		dev_scatTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_scatRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	}

	// Compute imaging condition at last sample its=nts-1
	kernel_exec(imagingAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_modelTomoOut, dev_scatLeft[iGpu], dev_tomoRecWavefieldIn, host_nts-1));

	// Copy slice at nts-1 to scattered wavefield
	if (saveWavefield == 1) {cuda_call(cudaMemcpy(dev_wavefield1Out+(host_nts-1)*host_nz*host_nx, dev_scatLeft[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));}

}

// Leg 2 adjoint [time]: s-> m <- i <- d
void computeTomoLeg2TimeAdj(double *dev_tomoSrcWavefieldDt2In, double *dev_tomoRecWavefieldIn, double *dev_modelTomoOut, double *dev_wavefield1Out, double *dev_extReflectivityIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, int saveWavefield){

	// Note: No need to initialize wavefield1out because we are doing memcpy

	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
	cuda_call(cudaMemset(dev_scatLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_scatRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Compute secondary source for its=nts-1
	int its = host_nts-1;
	int iExtMin, iExtMax;
    iExtMin = (-its)/2;
    iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
    iExtMax = (host_nts-1-its)/2;
    iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Add 1 for the strict inequality in the "for loop"
    kernel_exec(imagingTimeTomoAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_tomoRecWavefieldIn, dev_ssRight[iGpu], dev_extReflectivityIn, its, iExtMin, iExtMax)); // Apply extended FWD imaging condition

	// Start propagating scattered wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Compute secondary source for its
	    iExtMin = (-its)/2;
	    iExtMin = std::max(iExtMin, -host_hExt) + host_hExt;
	    iExtMax = (host_nts-1-its)/2;
	    iExtMax = std::min(iExtMax, host_hExt) + host_hExt + 1; // Add 1 for the strict inequality in the "for loop"
	    kernel_exec(imagingTimeTomoAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_tomoRecWavefieldIn, dev_ssLeft[iGpu], dev_extReflectivityIn, its, iExtMin, iExtMax)); // Apply extended FWD imaging condition

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step forward
			kernel_exec(stepAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source
			kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2+1));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate fine time slice to coarse time slice
			kernel_exec(interpFineToCoarseSlice<<<dimGridIn, dimBlockIn>>>(dev_scatLeft[iGpu], dev_scatRight[iGpu], dev_p0[iGpu], it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its+1
		kernel_exec(imagingAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_modelTomoOut, dev_scatRight[iGpu], dev_tomoSrcWavefieldDt2In, its+1));

		// Copy slice at its+1 to adjoint scattered wavefield
		if (saveWavefield == 1) {cuda_call(cudaMemcpy(dev_wavefield1Out+(its+1)*host_nz*host_nx, dev_scatRight[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));}

		// Switch pointers for secondary source
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));

		// Switch pointers scattered wavefield
		dev_scatTemp1[iGpu] = dev_scatRight[iGpu];
		dev_scatRight[iGpu] = dev_scatLeft[iGpu];
		dev_scatLeft[iGpu] = dev_scatTemp1[iGpu];
		dev_scatTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_scatLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));

	}

	// Compute imaging condition at last sample its=nts-1
	kernel_exec(imagingAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_modelTomoOut, dev_scatRight[iGpu], dev_tomoSrcWavefieldDt2In, 0));

	// Copy slice at its=0 to adjoint scattered wavefield
	if (saveWavefield == 1) {cuda_call(cudaMemcpy(dev_wavefield1Out, dev_scatRight[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));}

}

// Leg 1 adjoint [offset]: s -> i -> m <- d
void computeTomoLeg1OffsetAdj(double *dev_tomoSrcWavefieldDt2In, double *dev_tomoRecWavefieldIn, double *dev_modelTomoOut, double *dev_wavefield1Out, double *dev_extReflectivityIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, int saveWavefield){

	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
	cuda_call(cudaMemset(dev_scatLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_scatRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Compute secondary source for first coarse time index (its = 0)
	int its = 0;
	kernel_exec(imagingOffsetFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssLeft[iGpu], dev_tomoSrcWavefieldDt2In, its)); // Apply extended FWD imaging condition

	// Start propagating scattered wavefield
	for (int its = 0; its < host_nts-1; its++){

		// Compute secondary source for first coarse time index (its+1)
		kernel_exec(imagingOffsetFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_extReflectivityIn, dev_ssRight[iGpu], dev_tomoSrcWavefieldDt2In, its+1)); // Apply time-extended FWD imaging condition

		for (int it2 = 1; it2 < host_sub+1; it2++){

			// Step forward
			kernel_exec(stepFwdGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source sample itw-1
			kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2-1));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate fine time slice to coarse time slice
			kernel_exec(interpFineToCoarseSlice<<<dimGridIn, dimBlockIn>>>(dev_scatLeft[iGpu], dev_scatRight[iGpu], dev_p0[iGpu], it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its
		kernel_exec(imagingAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_modelTomoOut, dev_scatLeft[iGpu], dev_tomoRecWavefieldIn, its));

		// Copy slice at its to scattered wavefield
		if (saveWavefield == 1) {cuda_call(cudaMemcpy(dev_wavefield1Out+its*host_nz*host_nx, dev_scatLeft[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));}

		// Switch pointers for secondary source
		dev_ssTemp1[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

		// Switch pointers scattered wavefield
		dev_scatTemp1[iGpu] = dev_scatLeft[iGpu];
		dev_scatLeft[iGpu] = dev_scatRight[iGpu];
		dev_scatRight[iGpu] = dev_scatTemp1[iGpu];
		dev_scatTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_scatRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	}

	// Compute imaging condition at last sample its=nts-1
	kernel_exec(imagingAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_modelTomoOut, dev_scatLeft[iGpu], dev_tomoRecWavefieldIn, host_nts-1));

	// Copy slice at nts-1 to scattered wavefield
	if (saveWavefield == 1) {cuda_call(cudaMemcpy(dev_wavefield1Out+(host_nts-1)*host_nz*host_nx, dev_scatLeft[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));}

}

// Leg 2 adjoint [offset]: s-> m <- i <- d
void computeTomoLeg2OffsetAdj(double *dev_tomoSrcWavefieldDt2In, double *dev_tomoRecWavefieldIn, double *dev_modelTomoOut, double *dev_wavefield1Out, double *dev_extReflectivityIn, dim3 dimGridIn, dim3 dimBlockIn, int iGpu, int saveWavefield){

	// Initialize time slices on device
  	cuda_call(cudaMemset(dev_p0[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_p1[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_ssRight[iGpu], 0, host_nz*host_nx*sizeof(double)));
	cuda_call(cudaMemset(dev_scatLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));
  	cuda_call(cudaMemset(dev_scatRight[iGpu], 0, host_nz*host_nx*sizeof(double)));

	// Compute secondary source for its=nts-1
	int its = host_nts-1;
    kernel_exec(imagingOffsetTomoAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_tomoRecWavefieldIn, dev_ssRight[iGpu], dev_extReflectivityIn, its));

	// Start propagating scattered wavefield
	for (int its = host_nts-2; its > -1; its--){

		// Compute secondary source for its
	    kernel_exec(imagingOffsetTomoAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_tomoRecWavefieldIn, dev_ssLeft[iGpu], dev_extReflectivityIn, its));

		for (int it2 = host_sub-1; it2 > -1; it2--){

			// Step forward
			kernel_exec(stepAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu], dev_p0[iGpu], dev_vel2Dtw2[iGpu]));

			// Inject secondary source
			kernel_exec(injectSecondarySource<<<dimGridIn, dimBlockIn>>>(dev_ssLeft[iGpu], dev_ssRight[iGpu], dev_p0[iGpu], it2+1));

			// Damp wavefields
			kernel_exec(dampCosineEdge<<<dimGridIn, dimBlockIn>>>(dev_p0[iGpu], dev_p1[iGpu]));

			// Interpolate fine time slice to coarse time slice
			kernel_exec(interpFineToCoarseSlice<<<dimGridIn, dimBlockIn>>>(dev_scatLeft[iGpu], dev_scatRight[iGpu], dev_p0[iGpu], it2));

			// Switch pointers
			dev_temp1[iGpu] = dev_p0[iGpu];
			dev_p0[iGpu] = dev_p1[iGpu];
			dev_p1[iGpu] = dev_temp1[iGpu];
			dev_temp1[iGpu] = NULL;

		}

		// Apply imaging condition at its+1
		kernel_exec(imagingAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_modelTomoOut, dev_scatRight[iGpu], dev_tomoSrcWavefieldDt2In, its+1));

		// Copy slice at its+1 to adjoint scattered wavefield
		if (saveWavefield == 1) {cuda_call(cudaMemcpy(dev_wavefield1Out+(its+1)*host_nz*host_nx, dev_scatRight[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));}

		// Switch pointers for secondary source
		dev_ssTemp1[iGpu] = dev_ssRight[iGpu];
		dev_ssRight[iGpu] = dev_ssLeft[iGpu];
		dev_ssLeft[iGpu] = dev_ssTemp1[iGpu];
		dev_ssTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_ssLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));

		// Switch pointers scattered wavefield
		dev_scatTemp1[iGpu] = dev_scatRight[iGpu];
		dev_scatRight[iGpu] = dev_scatLeft[iGpu];
		dev_scatLeft[iGpu] = dev_scatTemp1[iGpu];
		dev_scatTemp1[iGpu] = NULL;
  		cuda_call(cudaMemset(dev_scatLeft[iGpu], 0, host_nz*host_nx*sizeof(double)));

	}

	// Compute imaging condition at last sample its=nts-1
	kernel_exec(imagingAdjGpu<<<dimGridIn, dimBlockIn>>>(dev_modelTomoOut, dev_scatRight[iGpu], dev_tomoSrcWavefieldDt2In, 0));

	// Copy slice at its=0 to adjoint scattered wavefield
	if (saveWavefield == 1) {cuda_call(cudaMemcpy(dev_wavefield1Out, dev_scatRight[iGpu], host_nz*host_nx*sizeof(double), cudaMemcpyDeviceToDevice));}

}

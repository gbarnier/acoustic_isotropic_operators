#include "varDeclare.h"
#include <stdio.h>

/****************************************************************************************/
/***************************************** Debug shit ***********************************/
/****************************************************************************************/
__global__ void copyToWavefield(double *wavefield, double *slice, int its){

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	int iGlobalWavefield = its * dev_nz * dev_nx + iGlobal;
	wavefield[iGlobalWavefield] += slice[iGlobal];
}

__global__ void sumData(double *data, double *data1, double *data2, int nData){

	int iTime = blockIdx.x * BLOCK_SIZE + threadIdx.x; // Time coordinates
	int iRec = blockIdx.y * BLOCK_SIZE + threadIdx.y; // Receiver coordinates
	int iData = dev_nts * iRec + iTime; // 1D array index for the model on the global memory
	if (iData < nData){
		data[iData] = data1[iData] + data2[iData];
	}
}

__global__ void sumModels(double *model, double *model1, double *model2){

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

	model[iGlobal] = model1[iGlobal] + model2[iGlobal];

}

/* Extract and interpolate data */
__global__ void recordInterpDataFine(double *dev_newTimeSlice, double *dev_signalOut, int itw, int *dev_receiversPositionReg) {
	int iThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (iThread < dev_nReceiversReg) {
		dev_signalOut[dev_ntw*iThread+itw] += dev_newTimeSlice[dev_receiversPositionReg[iThread]];
	}
}

__global__ void interpWavefieldDebug(double *dev_wavefield, double *dev_timeSlice, int its, int it2, double *dev_vel2Dtw2In) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int iGlobal = dev_nz * ixGlobal + izGlobal;
	int iGlobalWavefield = its * dev_nz * dev_nx + iGlobal;
	dev_wavefield[iGlobalWavefield] += dev_timeSlice[iGlobal] * dev_interpFilter[it2] * dev_vel2Dtw2In[iGlobal];
	dev_wavefield[iGlobalWavefield+dev_nz*dev_nx] += dev_timeSlice[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2] * dev_vel2Dtw2In[iGlobal];

}

__global__ void scaleReflectivityDebug(double *dev_model, double *dev_reflectivityScaleIn){

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	dev_model[iGlobal] *= dev_reflectivityScaleIn[iGlobal];
}

__global__ void copyValueScaleDebug(double *dev_wavefield, double *dev_timeSlice, double *dev_vel2Dtw2In, int its) {

	long long izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	long long ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	long long iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	long long iGlobalWavefield = its * dev_nz * dev_nx + iGlobal;
	dev_wavefield[iGlobalWavefield] = dev_timeSlice[iGlobal];// * dev_vel2Dtw2In[iGlobal];

}

/****************************************************************************************/
/***************************************** Injection ************************************/
/****************************************************************************************/
/* Inject source: no need for a "if" statement because the number of threads = nb devices */
__global__ void injectSource(double *dev_signalIn, double *dev_timeSlice, int itw, int *dev_sourcesPositionReg){
	int iThread = blockIdx.x * blockDim.x + threadIdx.x;
	dev_timeSlice[dev_sourcesPositionReg[iThread]] += dev_signalIn[iThread * dev_ntw + itw]; // Time is the fast axis
}

/* Interpolate and inject data */
__global__ void interpInjectData(double *dev_signalIn, double *dev_timeSlice, int its, int it2, int *dev_receiversPositionReg) {
	int iThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (iThread < dev_nReceiversReg) {
		dev_timeSlice[dev_receiversPositionReg[iThread]] += dev_signalIn[dev_nts*iThread+its] * dev_interpFilter[it2+1] + dev_signalIn[dev_nts*iThread+its+1] * dev_interpFilter[dev_hInterpFilter+it2+1];
	}
}

/* Interpolate and inject secondary source at fine time-sampling */
__global__ void injectSecondarySource(double *dev_ssLeft, double *dev_ssRight, double *dev_p0, int indexFilter){
	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	dev_p0[iGlobal] += dev_ssLeft[iGlobal] * dev_interpFilter[indexFilter] + dev_ssRight[iGlobal] * dev_interpFilter[dev_hInterpFilter+indexFilter];
}

/****************************************************************************************/
/*************************************** Extraction *************************************/
/****************************************************************************************/
/* Extract source */
__global__ void recordSource(double *dev_wavefield, double *dev_signalOut, int itw, int *dev_sourcesPositionReg) {
	int iThread = blockIdx.x * blockDim.x + threadIdx.x;
	dev_signalOut[dev_ntw*iThread + itw] += dev_wavefield[dev_sourcesPositionReg[iThread]];
}

/* Extract and interpolate data */
__global__ void recordInterpData(double *dev_newTimeSlice, double *dev_signalOut, int its, int it2, int *dev_receiversPositionReg) {

	int iThread = blockIdx.x * blockDim.x + threadIdx.x;
	if (iThread < dev_nReceiversReg) {
		// printf("dev_receiversPositionReg[iThread] = %d \n", dev_receiversPositionReg[iThread]);
		dev_signalOut[dev_nts*iThread+its]   += dev_newTimeSlice[dev_receiversPositionReg[iThread]] * dev_interpFilter[it2];
		dev_signalOut[dev_nts*iThread+its+1] += dev_newTimeSlice[dev_receiversPositionReg[iThread]] * dev_interpFilter[dev_hInterpFilter+it2];
	}
}

/****************************************************************************************/
/******************************** Wavefield extractions *********************************/
/****************************************************************************************/
__global__ void interpWavefield(double *dev_wavefield, double *dev_timeSlice, int its, int it2) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	int iGlobalWavefield = its * dev_nz * dev_nx + iGlobal;
	dev_wavefield[iGlobalWavefield] += dev_timeSlice[iGlobal] * dev_interpFilter[it2]; // its
	dev_wavefield[iGlobalWavefield+dev_nz*dev_nx] += dev_timeSlice[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1

}

__global__ void recordScaleWavefield(double *dev_wavefield, double *dev_timeSlice, int its, double *dev_reflectivityScale, double *dev_vel2Dtw2) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	long long iGlobalWavefield = its * dev_nz * dev_nx + iGlobal;

	dev_wavefield[iGlobalWavefield] += dev_timeSlice[iGlobal] * dev_reflectivityScale[iGlobal] * dev_vel2Dtw2[iGlobal];
}

__global__ void extractInterpAdjointWavefield(double *dev_timeSliceLeft, double *dev_timeSliceRight, double *dev_timeSliceFine, int it2) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	dev_timeSliceLeft[iGlobal]  += dev_timeSliceFine[iGlobal] * dev_interpFilter[it2]; // its
	dev_timeSliceRight[iGlobal] += dev_timeSliceFine[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1
}

__global__ void interpFineToCoarseSlice(double *dev_timeSliceLeft, double *dev_timeSliceRight, double *dev_timeSliceFine, int it2) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	dev_timeSliceLeft[iGlobal]  += dev_timeSliceFine[iGlobal] * dev_interpFilter[it2]; // its
	dev_timeSliceRight[iGlobal] += dev_timeSliceFine[iGlobal] * dev_interpFilter[dev_hInterpFilter+it2]; // its+1
}

/****************************************************************************************/
/************************************ Time derivative ***********************************/
/****************************************************************************************/

__global__ void srcWfldSecondTimeDerivative(double *dev_wavefield, double *dev_slice0, double *dev_slice1, double *dev_slice2, int its) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	int iGlobalWavefield = its * dev_nz * dev_nx + iGlobal;

	// Apply second time derivative
	dev_wavefield[iGlobalWavefield] = dev_cSide * ( dev_slice0[iGlobal] + dev_slice2[iGlobal] ) + dev_cCenter * dev_slice1[iGlobal];
}

/****************************************************************************************/
/************************************** Damping *****************************************/
/****************************************************************************************/
__global__ void dampCosineEdge(double *dev_p1, double *dev_p2) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

	// Compute distance to the closest edge of model
	int distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, dev_nz-izGlobal-1-FAT, dev_nx-ixGlobal-1-FAT);
	if (distToEdge < dev_minPad){

		// Compute damping coefficient
		double damp = dev_cosDampingCoeff[distToEdge];

		// Apply damping
		dev_p1[iGlobal] *= damp;
		dev_p2[iGlobal] *= damp;
	}
}

/****************************************************************************************/
/************************************** Scaling *****************************************/
/****************************************************************************************/
__global__ void scaleReflectivity(double *dev_model, double *dev_reflectivityScaleIn, double *dev_vel2Dtw2In){

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	dev_model[iGlobal] *= dev_vel2Dtw2In[iGlobal] * dev_reflectivityScaleIn[iGlobal];
}

__global__ void scaleReflectivityExt(double *dev_model, double *dev_reflectivityScale, double *dev_vel2Dtw2){

	int iz = FAT + blockIdx.x * BLOCK_SIZE_EXT + threadIdx.x; // z-coordinate
	int ix = FAT + blockIdx.y * BLOCK_SIZE_EXT + threadIdx.y; // x-coordinate
	int iSpace = dev_nz * ix + iz;
	int iExt = blockIdx.z * BLOCK_SIZE_EXT + threadIdx.z; // Extended axis coordinate
	int iModel = iExt * dev_nz * dev_nx + iSpace; // 1D array index for the model on the global memory

	if (iExt < dev_nExt){
		dev_model[iModel] *= dev_reflectivityScale[iSpace] * dev_vel2Dtw2[iSpace];
	}
}

/****************************************************************************************/
/************************************** Imaging *****************************************/
/****************************************************************************************/

// Non-extended
__global__ void imagingFwdGpu(double *dev_model, double *dev_timeSlice, int its, double *dev_sourceWavefieldDts) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	int iGlobalWavefield = its * dev_nz * dev_nx + iGlobal;
	dev_timeSlice[iGlobal] = dev_model[iGlobal] * dev_sourceWavefieldDts[iGlobalWavefield];
}

__global__ void imagingAdjGpu(double *dev_model, double *dev_timeSlice, double *dev_srcWavefieldDts, int its){

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	int iGlobalWavefield = its * dev_nz * dev_nx + iGlobal;
	dev_model[iGlobal] += dev_srcWavefieldDts[iGlobalWavefield] * dev_timeSlice[iGlobal];
}

// Time-lags
__global__ void imagingTimeFwdGpu(double *dev_model, double *dev_timeSlice, double *dev_srcWavefieldDts, int its, int iExtMin, int iExtMax){

	int iz = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ix = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iSpace = dev_nz * ix + iz; // 1D array index for the model on the global memory
	int iWavefield = its * dev_nz * dev_nx + iSpace; // Index for source wavefield at its

	for (int iExt = iExtMin; iExt < iExtMax; iExt++){

		int iModelExt = iExt * dev_nz * dev_nx + iSpace; // Compute index for extended model
		int iSrcWavefield = iWavefield - 2 * (iExt-dev_hExt) * dev_nz * dev_nx; // Compute index for source wavefield
		dev_timeSlice[iSpace] += dev_model[iModelExt] * dev_srcWavefieldDts[iSrcWavefield]; // Compute FWD imaging condition
	}
}

__global__ void imagingTimeAdjGpu(double *dev_model, double *dev_receiverTimeSlice, double *dev_srcWavefieldDts, int its, int iExtMin, int iExtMax){

	int iz = FAT + blockIdx.x * BLOCK_SIZE_EXT + threadIdx.x; // z-coordinate
	int ix = FAT + blockIdx.y * BLOCK_SIZE_EXT + threadIdx.y; // x-coordinate
	int iSpace = dev_nz * ix + iz; // 1D array index on spatial grid
	int iExt = iExtMin + blockIdx.z * BLOCK_SIZE_EXT + threadIdx.z; // Extended axis coordinate
	int iSrcWavefield = (its-2*(iExt-dev_hExt)) * dev_nz * dev_nx + iSpace; // Index for source wavefield at its
	int iModel = iExt * dev_nz * dev_nx + iSpace; // Extended model index

	if (iExt < iExtMax){
		dev_model[iModel] += dev_receiverTimeSlice[iSpace] * dev_srcWavefieldDts[iSrcWavefield]; // Try without +=
	}
}

// Subsurface offsets
__global__ void imagingOffsetFwdGpu(double *dev_model, double *dev_timeSlice, double *dev_srcWavefieldDts, int its){

	int iz = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // z-coordinate on main grid
	int ix = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // x-coordinate on main grid
	int iSpace = dev_nz * ix + iz; // 1D array index for the model on the global memory

	int iExtMin=max2(-dev_hExt, ix+1+FAT+dev_hExt-dev_nx);
	int iExtMax=min2(ix-dev_hExt-FAT, dev_hExt)+1;

	for (int iExt=iExtMin; iExt<iExtMax; iExt++){
		int iModel = dev_nz * dev_nx * (iExt+dev_hExt) + dev_nz * (ix-iExt) + iz; // model(iz, ix-iOffset, iOffset+hOffset)
		int iSrcWavefield = dev_nz * dev_nx * its + (ix-2*iExt) * dev_nz + iz; // src(iz, ix-2*iOffset, its)
		dev_timeSlice[iSpace] += dev_model[iModel] * dev_srcWavefieldDts[iSrcWavefield];
	}
}

__global__ void imagingOffsetAdjGpu(double *dev_model, double *dev_timeSlice, double *dev_srcWavefieldDts, int its){

	int iz = FAT + blockIdx.x * BLOCK_SIZE_EXT + threadIdx.x; // z-coordinate on main grid
	int ix = FAT + dev_hExt + blockIdx.y * BLOCK_SIZE_EXT + threadIdx.y; // x-coordinate on main grid for the model where we evaluate the image
	int iExt = blockIdx.z * BLOCK_SIZE_EXT + threadIdx.z; // offset coordinate (iOffset = 0, ..., dev_nOffset-1)

	if ( (ix < dev_nx-FAT-dev_hExt) && (iExt < dev_nExt) ){
		int iExtShift=iExt-dev_hExt;
		int iModel = dev_nz * dev_nx * iExt + dev_nz * ix + iz; // Model index
		int iSrcWavefield = dev_nz * dev_nx * its + dev_nz * (ix-iExtShift) + iz; // Source wavefield index
		int iRecWavefield = dev_nz * (ix+iExtShift) + iz; // Receiver wavefield index
		dev_model[iModel] += dev_timeSlice[iRecWavefield] * dev_srcWavefieldDts[iSrcWavefield];
	}
}

// Non-extended
__global__ void imagingAdjTomoGpu(double *dev_wavefieldIn, double *dev_timeSliceOut, double *dev_extReflectivityIn, int its) {

	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory
	int iGlobalWavefield = its * dev_nz * dev_nx + iGlobal;
	dev_timeSliceOut[iGlobal] = dev_extReflectivityIn[iGlobal] * dev_wavefieldIn[iGlobalWavefield];
}

__global__ void imagingTimeTomoAdjGpu(double *dev_wavefieldIn, double *dev_timeSliceOut, double *dev_extReflectivityIn, int its, int iExtMin, int iExtMax) {

	int iz = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ix = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int iSpace = dev_nz * ix + iz; // 1D array index for the model on the global memory
	int iWavefield = its * dev_nz * dev_nx + iSpace; // Index for source wavefield at its

	for (int iExt = iExtMin; iExt < iExtMax; iExt++){

		int iModelExt = iExt * dev_nz * dev_nx + iSpace; // Compute index for extended model
		int iRecWavefield = iWavefield + 2 * (iExt-dev_hExt) * dev_nz * dev_nx; // Compute index for source wavefield
		dev_timeSliceOut[iSpace] += dev_extReflectivityIn[iModelExt] * dev_wavefieldIn[iRecWavefield];
	}
}


/****************************************************************************************/
/*********************************** Forward steppers ***********************************/
/****************************************************************************************/
/* Forward stepper (no damping) */
__global__ void stepFwdGpu(double *dev_o, double *dev_c, double *dev_n, double *dev_vel2Dtw2) {

	__shared__ double shared_c[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // Allocate shared memory
	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int izLocal = FAT + threadIdx.x; // z-coordinate on the shared grid
	int ixLocal = FAT + threadIdx.y; // x-coordinate on the shared grid
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

	// Copy current slice from global to shared memory
	shared_c[ixLocal][izLocal] = dev_c[iGlobal];

	// Copy current slice from global to shared -- edges
	if (threadIdx.y < FAT) {
		shared_c[ixLocal-FAT][izLocal] = dev_c[iGlobal-dev_nz*FAT]; // Left side
		shared_c[ixLocal+BLOCK_SIZE][izLocal] = dev_c[iGlobal+dev_nz*BLOCK_SIZE] ; // Right side
	}
	if (threadIdx.x < FAT) {
		shared_c[ixLocal][izLocal-FAT] = dev_c[iGlobal-FAT]; // Up
		shared_c[ixLocal][izLocal+BLOCK_SIZE] = dev_c[iGlobal+BLOCK_SIZE]; // Down
	}
	__syncthreads(); // Synchronise all threads within each block -- look new sync options

	dev_n[iGlobal] =  dev_vel2Dtw2[iGlobal] * ( dev_zCoeff[0] * shared_c[ixLocal][izLocal]
				   +  dev_zCoeff[1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
				   +  dev_zCoeff[2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
				   +  dev_zCoeff[3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
				   +  dev_zCoeff[4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
				   +  dev_zCoeff[5] * ( shared_c[ixLocal][izLocal-5] + shared_c[ixLocal][izLocal+5] )
				   +  dev_xCoeff[0] * shared_c[ixLocal][izLocal]
				   +  dev_xCoeff[1] * ( shared_c[ixLocal+1][izLocal] + shared_c[ixLocal-1][izLocal] )
				   +  dev_xCoeff[2] * ( shared_c[ixLocal+2][izLocal] + shared_c[ixLocal-2][izLocal] )
				   +  dev_xCoeff[3] * ( shared_c[ixLocal+3][izLocal] + shared_c[ixLocal-3][izLocal] )
				   +  dev_xCoeff[4] * ( shared_c[ixLocal+4][izLocal] + shared_c[ixLocal-4][izLocal] )
				   +  dev_xCoeff[5] * ( shared_c[ixLocal+5][izLocal] + shared_c[ixLocal-5][izLocal] ) )
				   +  shared_c[ixLocal][izLocal] + shared_c[ixLocal][izLocal] - dev_o[iGlobal];
}

/****************************************************************************************/
/*********************************** Adjoint steppers ***********************************/
/****************************************************************************************/

/* Adjoint stepper (no damping) */
__global__ void stepAdjGpu(double *dev_o, double *dev_c, double *dev_n, double *dev_vel2Dtw2) {

	__shared__ double shared_c[BLOCK_SIZE+2*FAT][BLOCK_SIZE+2*FAT]; // Allocate shared memory
	int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
	int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
	int izLocal = FAT + threadIdx.x; // z-coordinate on the shared grid
	int ixLocal = FAT + threadIdx.y; // z-coordinate on the shared grid
	int iGlobal = dev_nz * ixGlobal + izGlobal; // 1D array index for the model on the global memory

	// Copy current slice from global memory to shared memory
	shared_c[ixLocal][izLocal] = dev_c[iGlobal] * dev_vel2Dtw2[iGlobal];

	// Copy current slice from global memory to shared -- edges ("halo")
	if (threadIdx.y < FAT) {
		shared_c[ixLocal-FAT][izLocal] = dev_c[iGlobal-dev_nz*FAT] * dev_vel2Dtw2[iGlobal-dev_nz*FAT]; // Left side
		shared_c[ixLocal+BLOCK_SIZE][izLocal] = dev_c[iGlobal+dev_nz*BLOCK_SIZE] * dev_vel2Dtw2[iGlobal+dev_nz*BLOCK_SIZE]; // Right side
	}
	if (threadIdx.x < FAT) {
		shared_c[ixLocal][izLocal-FAT] = dev_c[iGlobal-FAT] * dev_vel2Dtw2[iGlobal-FAT]; // Up
		shared_c[ixLocal][izLocal+BLOCK_SIZE] = dev_c[iGlobal+BLOCK_SIZE] * dev_vel2Dtw2[iGlobal+BLOCK_SIZE]; // Down
	}
	__syncthreads(); // Synchronise all threads within each block

	dev_o[iGlobal] =  ( dev_zCoeff[0] * shared_c[ixLocal][izLocal]
				   +  dev_zCoeff[1] * ( shared_c[ixLocal][izLocal-1] + shared_c[ixLocal][izLocal+1] )
				   +  dev_zCoeff[2] * ( shared_c[ixLocal][izLocal-2] + shared_c[ixLocal][izLocal+2] )
				   +  dev_zCoeff[3] * ( shared_c[ixLocal][izLocal-3] + shared_c[ixLocal][izLocal+3] )
				   +  dev_zCoeff[4] * ( shared_c[ixLocal][izLocal-4] + shared_c[ixLocal][izLocal+4] )
				   +  dev_zCoeff[5] * ( shared_c[ixLocal][izLocal-5] + shared_c[ixLocal][izLocal+5] )
				   +  dev_xCoeff[0] * shared_c[ixLocal][izLocal]
				   +  dev_xCoeff[1] * ( shared_c[ixLocal+1][izLocal] + shared_c[ixLocal-1][izLocal] )
				   +  dev_xCoeff[2] * ( shared_c[ixLocal+2][izLocal] + shared_c[ixLocal-2][izLocal] )
				   +  dev_xCoeff[3] * ( shared_c[ixLocal+3][izLocal] + shared_c[ixLocal-3][izLocal] )
				   +  dev_xCoeff[4] * ( shared_c[ixLocal+4][izLocal] + shared_c[ixLocal-4][izLocal] )
				   +  dev_xCoeff[5] * ( shared_c[ixLocal+5][izLocal] + shared_c[ixLocal-5][izLocal] ) )
				   +  dev_c[iGlobal] + dev_c[iGlobal] - dev_n[iGlobal];
}

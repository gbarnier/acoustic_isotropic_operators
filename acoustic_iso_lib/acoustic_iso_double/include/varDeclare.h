#ifndef VAR_DECLARE_H
#define VAR_DECLARE_H 1

#include <math.h>
#define BLOCK_SIZE 16
#define BLOCK_SIZE_DATA 128
#define BLOCK_SIZE_EXT 8
#define FAT 5
#define COEFF_SIZE 6 // Laplacian coefficient array for 10th order
#define PI_CUDA M_PI // Import the number "Pi" from the math library
#define PAD_MAX 200 // Maximum number of points for padding (on one side)
#define SUB_MAX 100 // Maximum subsampling value

#define min2(v1,v2) (((v1)<(v2))?(v1):(v2)) /* Minimum function */
#define max2(v1,v2) (((v1)>(v2))?(v1):(v2)) /* Minimum function */

/************************************* DEVICE DECLARATION *******************************/
// Device function
__device__ int min4(int v1,int v2,int v3,int v4){return min2(min2(v1,v2),min2(v3,v4));}

// Constant memory variables
__constant__ double dev_zCoeff[COEFF_SIZE]; // 10th-order Laplacian coefficients on Device
__constant__ double dev_xCoeff[COEFF_SIZE];

__constant__ int dev_nInterpFilter; // Time interpolation filter length
__constant__ int dev_hInterpFilter; // Time interpolation filter half-length
__constant__ double dev_interpFilter[2*(SUB_MAX+1)]; // Time interpolation filter stored in constant memory

__constant__ int dev_nts; // Number of time steps at the coarse time sampling on Device
__constant__ int dev_ntw; // Number of time steps at the fine time sampling on Device
__constant__ int dev_nz; // nz on Device
__constant__ int dev_nx; // nx on Device
__constant__ int dev_sub; // Subsampling in time
__constant__ int dev_nExt; // Length of extension axis
__constant__ int dev_hExt; // Half-length of extension axis

__constant__ int dev_nSourcesReg; // Nb of source grid points
__constant__ int dev_nReceiversReg; // Nb of receiver grid points

__constant__ double dev_alphaCos; // Decay coefficient
__constant__ int dev_minPad; // Minimum padding length
__constant__ double dev_cosDampingCoeff[PAD_MAX]; // Padding array
__constant__ double dev_cSide;
__constant__ double dev_cCenter;

// Global memory variables
int **dev_sourcesPositionReg; // Array containing the positions of the sources on the regular grid
int **dev_receiversPositionReg; // Array containing the positions of the receivers on the regular grid
double **dev_p0, **dev_p1, **dev_temp1; // Temporary slices for stepping
double **dev_ss0, **dev_ss1, **dev_ss2, **dev_ssTemp2;
double **dev_ssLeft, **dev_ssRight, **dev_ssTemp1; // Temporary slices for secondary source
double **dev_scatLeft, **dev_scatRight, **dev_scatTemp1; // Temporary slices for scattered wavefield (used in tomo)
double **dev_sourcesSignals; // Sources for modeling
double **dev_vel2Dtw2; // Precomputed scaling v^2 * dtw^2
double **dev_reflectivityScale; // scale = -2.0 / (vel*vel*vel)

// Nonlinear modeling
double **dev_modelRegDtw; // Model for nonlinear propagation (wavelet)
double **dev_dataRegDts; // Data on device at coarse time-sampling (converted to regular grid)
double *dev_wavefieldDts; // Source wavefield

// Born
double **dev_modelBorn, **dev_modelBornExt; // Reflectivity model for Born / Born extended
double **dev_BornSrcWavefield, *dev_BornSecWavefield;

// Tomo
double **dev_modelTomo;  // Model for tomo
double **dev_extReflectivity; // Extended reflectivity for tomo
double **dev_tomoSrcWavefieldDt2, **dev_tomoSecWavefield1, **dev_tomoSecWavefield2;

// Wemva
double **dev_modelWemva;
double **dev_wemvaDataRegDts; // Seismic data for Wemva operator
double **dev_wemvaExtImage; // Output of Wemva forward
double **dev_wemvaSrcWavefieldDt2, **dev_wemvaSecWavefield1, **dev_wemvaSecWavefield2;

// Streams
cudaStream_t *stream1, *stream2;

/************************************* HOST DECLARATION *********************************/
long long host_nz; // Includes padding + FAT
long long host_nx;
double host_dz;
double host_dx;
int host_nts;
double host_dts;
int host_ntw;
int host_sub;
int host_nExt; // Length of extended axis
int host_hExt; // Half-length of extended axis
double host_cSide, host_cCenter; // Coefficients for the second-order time derivative
int host_leg1, host_leg2; // Flags to indicate which legs to compute in tomo and wemva

#endif

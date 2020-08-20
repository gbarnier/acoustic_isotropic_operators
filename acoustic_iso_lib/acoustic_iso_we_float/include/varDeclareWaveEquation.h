#ifndef VAR_DECLARE_WAVE_EQUATION_H
#define VAR_DECLARE_WAVE_EQUATION_H 1

#include <math.h>
#define BLOCK_SIZE 8
#define FAT 5
#define COEFF_SIZE 6
#define PAD_MAX 200 // Maximum number of points for padding (on one side)


#define min2(v1,v2) (((v1)<(v2))?(v1):(v2)) /* Minimum function */
#define max2(v1,v2) (((v1)>(v2))?(v1):(v2)) /* Minimum function */
/*************************** Constant memory variable *************************/
// Device function
__device__ int min4(int v1,int v2,int v3,int v4){return min2(min2(v1,v2),min2(v3,v4));}

__constant__ float dev_zCoeff[COEFF_SIZE]; // 8th-order Laplacian coefficients on Device
__constant__ float dev_xCoeff[COEFF_SIZE];

__constant__ int dev_nts; // Number of time steps at the coarse time sampling on Device
__constant__ int dev_nz; // nz on Device
__constant__ int dev_nx; // nx on Device
__constant__ float dev_dx; // dx on Device
__constant__ float dev_dz; // dz on Device
__constant__ float dev_dts; // dz on Device
__constant__ float dev_cosDampingCoeff[PAD_MAX]; // Padding array
__constant__ int dev_minPad; // Minimum padding length

/****************************** Device Pointers *******************************/
float **dev_p0, **dev_p1;
float **dev_slsqDt2; // Precomputed scaling slsq/dts^2 x
float **dev_cosDamp; // Precomputed scaling slsq/dts^2 x

/************************************* HOST DECLARATION *********************************/
long long host_nz; // Includes padding + FAT
long long host_nx;
float host_dz;
float host_dx;
int host_nts;
float host_dts;

#endif


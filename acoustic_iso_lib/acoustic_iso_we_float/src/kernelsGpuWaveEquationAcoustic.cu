#include "varDeclareWaveEquation.h"
#include <stdio.h>

/****************************************************************************************/
/*********************************** Forward kernel ***********************************/
/****************************************************************************************/
/* kernel to compute forward wabe equation */
__global__ void ker_we_fwd(float* dev_p0, float* dev_p1, float* dev_slsqDt2, float *dev_cosDamp, int absoluteFirstTimeSampleForBlock, int absoluteLastTimeSampleForBlock){
                             //float* dev_c_all,float* dev_n_all, float* dev_elastic_param_scaled) {

    // calculate global and local x/z/t coordinates
    int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
    int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
    int itInTimeBlock = blockIdx.z * BLOCK_SIZE + threadIdx.z;
    int itGlobal = absoluteFirstTimeSampleForBlock + itInTimeBlock;
    int iGlobal = dev_nz*ixGlobal + izGlobal;

    int iGlobal_prev = dev_nz*dev_nx*(itInTimeBlock-1) + dev_nz*ixGlobal + izGlobal;
    int iGlobal_cur = dev_nz*dev_nx*itInTimeBlock + dev_nz*ixGlobal + izGlobal;
    int iGlobal_next = dev_nz*dev_nx*(itInTimeBlock+1) + dev_nz*ixGlobal + izGlobal;

    //t=0 boundary condition.
    if(itGlobal==0){
      dev_p0[iGlobal_cur] += dev_slsqDt2[iGlobal]*(dev_cosDamp[iGlobal]*-2*dev_p1[iGlobal_cur]+dev_p1[iGlobal_next])  \
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_xCoeff[1]*(dev_p1[iGlobal_cur+1*dev_nz]+dev_p1[iGlobal_cur-1*dev_nz])+ \
                              	   dev_xCoeff[2]*(dev_p1[iGlobal_cur+2*dev_nz]+dev_p1[iGlobal_cur-2*dev_nz])+ \
                              	   dev_xCoeff[3]*(dev_p1[iGlobal_cur+3*dev_nz]+dev_p1[iGlobal_cur-3*dev_nz]) + \
                              	   dev_xCoeff[4]*(dev_p1[iGlobal_cur+4*dev_nz]+dev_p1[iGlobal_cur-4*dev_nz]) + \
                              	   dev_xCoeff[5]*(dev_p1[iGlobal_cur+5*dev_nz]+dev_p1[iGlobal_cur-5*dev_nz]) + \
                                   dev_zCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_zCoeff[1]*(dev_p1[iGlobal_cur+1]+dev_p1[iGlobal_cur-1])+ \
                              	   dev_zCoeff[2]*(dev_p1[iGlobal_cur+2]+dev_p1[iGlobal_cur-2])+ \
                              	   dev_zCoeff[3]*(dev_p1[iGlobal_cur+3]+dev_p1[iGlobal_cur-3]) + \
                              	   dev_zCoeff[4]*(dev_p1[iGlobal_cur+4]+dev_p1[iGlobal_cur-4]) + \
                              	   dev_zCoeff[5]*(dev_p1[iGlobal_cur+5]+dev_p1[iGlobal_cur-5])*dev_cosDamp[iGlobal]);
    }
    else if(itInTimeBlock==0){
      //do nothing in this case.
    }
    //t=nt-1 boundary condition
    else if(itGlobal==dev_nts-1){
      dev_p0[iGlobal_cur] += dev_slsqDt2[iGlobal]*dev_cosDamp[iGlobal]*(dev_p1[iGlobal_prev]-2*dev_p1[iGlobal_cur]) \
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_xCoeff[1]*(dev_p1[iGlobal_cur+1*dev_nz]+dev_p1[iGlobal_cur-1*dev_nz])+ \
                              	   dev_xCoeff[2]*(dev_p1[iGlobal_cur+2*dev_nz]+dev_p1[iGlobal_cur-2*dev_nz])+ \
                              	   dev_xCoeff[3]*(dev_p1[iGlobal_cur+3*dev_nz]+dev_p1[iGlobal_cur-3*dev_nz]) + \
                              	   dev_xCoeff[4]*(dev_p1[iGlobal_cur+4*dev_nz]+dev_p1[iGlobal_cur-4*dev_nz]) + \
                              	   dev_xCoeff[5]*(dev_p1[iGlobal_cur+5*dev_nz]+dev_p1[iGlobal_cur-5*dev_nz]) + \
                                   dev_zCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_zCoeff[1]*(dev_p1[iGlobal_cur+1]+dev_p1[iGlobal_cur-1])+ \
                              	   dev_zCoeff[2]*(dev_p1[iGlobal_cur+2]+dev_p1[iGlobal_cur-2])+ \
                              	   dev_zCoeff[3]*(dev_p1[iGlobal_cur+3]+dev_p1[iGlobal_cur-3]) + \
                              	   dev_zCoeff[4]*(dev_p1[iGlobal_cur+4]+dev_p1[iGlobal_cur-4]) + \
                              	   dev_zCoeff[5]*(dev_p1[iGlobal_cur+5]+dev_p1[iGlobal_cur-5]))*dev_cosDamp[iGlobal];
    }
    else if(itGlobal<absoluteLastTimeSampleForBlock){
      dev_p0[iGlobal_cur] += dev_slsqDt2[iGlobal]*(dev_cosDamp[iGlobal]*dev_p1[iGlobal_prev] -dev_cosDamp[iGlobal]*2*dev_p1[iGlobal_cur] + dev_p1[iGlobal_next]) \
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_xCoeff[1]*(dev_p1[iGlobal_cur+1*dev_nz]+dev_p1[iGlobal_cur-1*dev_nz])+ \
                              	   dev_xCoeff[2]*(dev_p1[iGlobal_cur+2*dev_nz]+dev_p1[iGlobal_cur-2*dev_nz])+ \
                              	   dev_xCoeff[3]*(dev_p1[iGlobal_cur+3*dev_nz]+dev_p1[iGlobal_cur-3*dev_nz]) + \
                              	   dev_xCoeff[4]*(dev_p1[iGlobal_cur+4*dev_nz]+dev_p1[iGlobal_cur-4*dev_nz]) + \
                              	   dev_xCoeff[5]*(dev_p1[iGlobal_cur+5*dev_nz]+dev_p1[iGlobal_cur-5*dev_nz]) + \
                                   dev_zCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_zCoeff[1]*(dev_p1[iGlobal_cur+1]+dev_p1[iGlobal_cur-1])+ \
                              	   dev_zCoeff[2]*(dev_p1[iGlobal_cur+2]+dev_p1[iGlobal_cur-2])+ \
                              	   dev_zCoeff[3]*(dev_p1[iGlobal_cur+3]+dev_p1[iGlobal_cur-3]) + \
                              	   dev_zCoeff[4]*(dev_p1[iGlobal_cur+4]+dev_p1[iGlobal_cur-4]) + \
                              	   dev_zCoeff[5]*(dev_p1[iGlobal_cur+5]+dev_p1[iGlobal_cur-5])*dev_cosDamp[iGlobal]);
    }

}

/* kernel to compute adjoint time step */
__global__ void ker_we_adj(float* dev_p0, float* dev_p1,
                             float* dev_slsqDt2,float *dev_cosDamp,
                             int absoluteFirstTimeSampleForBlock, int absoluteLastTimeSampleForBlock){
                             //float* dev_c_all,float* dev_n_all, float* dev_elastic_param_scaled) {
    // calculate global and local x/z/t coordinates
    int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
    int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
    int itInTimeBlock = blockIdx.z * BLOCK_SIZE + threadIdx.z;
    int itGlobal = absoluteFirstTimeSampleForBlock + itInTimeBlock;
    int iGlobal = dev_nz*ixGlobal + izGlobal;

    int iGlobal_prev = dev_nz*dev_nx*(itInTimeBlock-1) + dev_nz*ixGlobal + izGlobal;
    int iGlobal_cur = dev_nz*dev_nx*itInTimeBlock + dev_nz*ixGlobal + izGlobal;
    int iGlobal_next = dev_nz*dev_nx*(itInTimeBlock+1) + dev_nz*ixGlobal + izGlobal;


    //t=0 boundary condition.
    if(itGlobal==0){
      dev_p0[iGlobal_cur] += dev_slsqDt2[iGlobal]*dev_cosDamp[iGlobal]*(-2*dev_p1[iGlobal_cur]+dev_p1[iGlobal_next])  \
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_xCoeff[1]*(dev_p1[iGlobal_cur+1*dev_nz]+dev_p1[iGlobal_cur-1*dev_nz])+ \
                              	   dev_xCoeff[2]*(dev_p1[iGlobal_cur+2*dev_nz]+dev_p1[iGlobal_cur-2*dev_nz])+ \
                              	   dev_xCoeff[3]*(dev_p1[iGlobal_cur+3*dev_nz]+dev_p1[iGlobal_cur-3*dev_nz]) + \
                              	   dev_xCoeff[4]*(dev_p1[iGlobal_cur+4*dev_nz]+dev_p1[iGlobal_cur-4*dev_nz]) + \
                              	   dev_xCoeff[5]*(dev_p1[iGlobal_cur+5*dev_nz]+dev_p1[iGlobal_cur-5*dev_nz]) + \
                                   dev_zCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_zCoeff[1]*(dev_p1[iGlobal_cur+1]+dev_p1[iGlobal_cur-1])+ \
                              	   dev_zCoeff[2]*(dev_p1[iGlobal_cur+2]+dev_p1[iGlobal_cur-2])+ \
                              	   dev_zCoeff[3]*(dev_p1[iGlobal_cur+3]+dev_p1[iGlobal_cur-3]) + \
                              	   dev_zCoeff[4]*(dev_p1[iGlobal_cur+4]+dev_p1[iGlobal_cur-4]) + \
                              	   dev_zCoeff[5]*(dev_p1[iGlobal_cur+5]+dev_p1[iGlobal_cur-5])*dev_cosDamp[iGlobal]);
    }
    else if(itInTimeBlock==0){
      //do nothing in this case.
    }
    //t=nt-1 boundary condition
    else if(itGlobal==dev_nts-1){
      dev_p0[iGlobal_cur] += dev_slsqDt2[iGlobal]*(dev_p1[iGlobal_prev]-2*dev_cosDamp[iGlobal]*dev_p1[iGlobal_cur]) \
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_xCoeff[1]*(dev_p1[iGlobal_cur+1*dev_nz]+dev_p1[iGlobal_cur-1*dev_nz])+ \
                              	   dev_xCoeff[2]*(dev_p1[iGlobal_cur+2*dev_nz]+dev_p1[iGlobal_cur-2*dev_nz])+ \
                              	   dev_xCoeff[3]*(dev_p1[iGlobal_cur+3*dev_nz]+dev_p1[iGlobal_cur-3*dev_nz]) + \
                              	   dev_xCoeff[4]*(dev_p1[iGlobal_cur+4*dev_nz]+dev_p1[iGlobal_cur-4*dev_nz]) + \
                              	   dev_xCoeff[5]*(dev_p1[iGlobal_cur+5*dev_nz]+dev_p1[iGlobal_cur-5*dev_nz]) + \
                                   dev_zCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_zCoeff[1]*(dev_p1[iGlobal_cur+1]+dev_p1[iGlobal_cur-1])+ \
                              	   dev_zCoeff[2]*(dev_p1[iGlobal_cur+2]+dev_p1[iGlobal_cur-2])+ \
                              	   dev_zCoeff[3]*(dev_p1[iGlobal_cur+3]+dev_p1[iGlobal_cur-3]) + \
                              	   dev_zCoeff[4]*(dev_p1[iGlobal_cur+4]+dev_p1[iGlobal_cur-4]) + \
                              	   dev_zCoeff[5]*(dev_p1[iGlobal_cur+5]+dev_p1[iGlobal_cur-5]))*dev_cosDamp[iGlobal];
    }
    else if(itGlobal<absoluteLastTimeSampleForBlock){
      dev_p0[iGlobal_cur] += dev_slsqDt2[iGlobal]*(dev_p1[iGlobal_prev] - dev_cosDamp[iGlobal]*2*dev_p1[iGlobal_cur] + dev_cosDamp[iGlobal]*dev_p1[iGlobal_next]) \
                                - (dev_xCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_xCoeff[1]*(dev_p1[iGlobal_cur+1*dev_nz]+dev_p1[iGlobal_cur-1*dev_nz])+ \
                              	   dev_xCoeff[2]*(dev_p1[iGlobal_cur+2*dev_nz]+dev_p1[iGlobal_cur-2*dev_nz])+ \
                              	   dev_xCoeff[3]*(dev_p1[iGlobal_cur+3*dev_nz]+dev_p1[iGlobal_cur-3*dev_nz]) + \
                              	   dev_xCoeff[4]*(dev_p1[iGlobal_cur+4*dev_nz]+dev_p1[iGlobal_cur-4*dev_nz]) + \
                              	   dev_xCoeff[5]*(dev_p1[iGlobal_cur+5*dev_nz]+dev_p1[iGlobal_cur-5*dev_nz]) + \
                                   dev_zCoeff[0]*(dev_p1[iGlobal_cur])+ \
                              	   dev_zCoeff[1]*(dev_p1[iGlobal_cur+1]+dev_p1[iGlobal_cur-1])+ \
                              	   dev_zCoeff[2]*(dev_p1[iGlobal_cur+2]+dev_p1[iGlobal_cur-2])+ \
                              	   dev_zCoeff[3]*(dev_p1[iGlobal_cur+3]+dev_p1[iGlobal_cur-3]) + \
                              	   dev_zCoeff[4]*(dev_p1[iGlobal_cur+4]+dev_p1[iGlobal_cur-4]) + \
                              	   dev_zCoeff[5]*(dev_p1[iGlobal_cur+5]+dev_p1[iGlobal_cur-5])*dev_cosDamp[iGlobal]);
    }
}

/****************************************************************************************/
/************************************** Damping *****************************************/
/****************************************************************************************/
__global__ void dampCosineEdge(float *dev_p1, int absoluteFirstTimeSampleForBlock, int absoluteLastTimeSampleForBlock){
    int izGlobal = FAT + blockIdx.x * BLOCK_SIZE + threadIdx.x; // Global z-coordinate
    int ixGlobal = FAT + blockIdx.y * BLOCK_SIZE + threadIdx.y; // Global x-coordinate
    int itInTimeBlock = blockIdx.z * BLOCK_SIZE + threadIdx.z;
    int itGlobal = absoluteFirstTimeSampleForBlock + itInTimeBlock;
    //int iGlobal = dev_nz*ixGlobal + izGlobal;

    int iGlobal_cur = dev_nz*dev_nx*itInTimeBlock + dev_nz*ixGlobal + izGlobal;
    //if first or last time sample zero out everywhere
    if(itGlobal==0 || itGlobal==dev_nts-1){
		dev_p1[iGlobal_cur] *= 0;
    }
    //else if( itGlobal<absoluteLastTimeSampleForBlock) {
    ////else dampen if withint boundary only 
    //    // Compute distance to the closest edge of model
    //    int distToEdge = min4(izGlobal-FAT, ixGlobal-FAT, dev_nz-izGlobal-1-FAT, dev_nx-ixGlobal-1-FAT);
    //    if (distToEdge < FAT ){
    //    	// Apply damping
    //    	dev_p1[iGlobal_cur] *= 0;
    //    }
    //    else if (distToEdge < dev_minPad){

    //    	// Compute damping coefficient
    //    	float damp = dev_cosDampingCoeff[distToEdge];

    //    	// Apply damping
    //    	dev_p1[iGlobal_cur] *= damp;
    //    }
    //}
}

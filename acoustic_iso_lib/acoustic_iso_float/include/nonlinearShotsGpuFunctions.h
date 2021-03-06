#ifndef NONLINEAR_SHOTS_GPU_FUNCTIONS_H
#define NONLINEAR_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/************************* Set GPU propagation parameters *********************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumber);
void initNonlinearGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateNonlinearGpu(float *vel2Dtw2, int iGpu, int iGpuId);
void deallocateNonlinearGpu(int iGpu, int iGpuId);

/************************* Nonlinear forward propagation **********************/
void propShotsFwdGpu(float *modelRegDtw, float *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);
void propShotsFwdGpuWavefield(float *modelRegDtw, float *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);

void propShotsFwdFsGpu(float *modelRegDtw, float *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);
void propShotsFwdFsGpuWavefield(float *modelRegDtw, float *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);

/************************* Nonlinear adjoint propagation **********************/
/* Adjoint propagation -- Data recorded at fine scale */
void propShotsAdjGpu(float *modelRegDtw, float *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);
void propShotsAdjGpuWavefield(float *modelRegDtw, float *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);

void propShotsAdjFsGpu(float *modelRegDtw, float *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);
void propShotsAdjFsGpuWavefield(float *modelRegDtw, float *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu, int iGpuId);

#endif

#ifndef NONLINEAR_SHOTS_GPU_FUNCTIONS_H
#define NONLINEAR_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/************************* Set GPU propagation parameters *********************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumber);
void initNonlinearGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateNonlinearGpu(double *vel2Dtw2, int iGpu, int iGpuId);
void deallocateNonlinearGpu(int iGpu, int iGpuId);

/************************* Nonlinear forward propagation **********************/
void propShotsFwdGpu(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);
void propShotsFwdGpuWavefield(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);

void propShotsFwdFsGpu(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);
void propShotsFwdFsGpuWavefield(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);

/************************* Nonlinear adjoint propagation **********************/
/* Adjoint propagation -- Data recorded at fine scale */
void propShotsAdjGpu(double *modelRegDtw, double *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);
void propShotsAdjGpuWavefield(double *modelRegDtw, double *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);

void propShotsAdjFsGpu(double *modelRegDtw, double *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);
void propShotsAdjFsGpuWavefield(double *modelRegDtw, double *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu, int iGpuId);

#endif

#ifndef NONLINEAR_SHOTS_GPU_FUNCTIONS_H
#define NONLINEAR_SHOTS_GPU_FUNCTIONS_H 1

/*********************************** Initialization **************************************/
bool getGpuInfo(int nGpu, int info, int deviceNumber);
void initNonlinearGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpu);
void allocateNonlinearGpu(float *vel2Dtw2, int iGpu);
void deallocateNonlinearGpu(int iGpu);

/*********************************** Nonlinear FWD **************************************/
void propShotsFwdGpu(float *modelRegDtw, float *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu);
void propShotsFwdGpuWavefield(float *modelRegDtw, float *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu);

/*********************************** Nonlinear ADJ **************************************/
/* Adjoint propagation -- Data recorded at fine scale */
void propShotsAdjGpu(float *modelRegDtw, float *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu);
void propShotsAdjGpuWavefield(float *modelRegDtw, float *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *wavefieldDts, int iGpu);

#endif

#ifndef NONLINEAR_SHOTS_GPU_FUNCTIONS_H
#define NONLINEAR_SHOTS_GPU_FUNCTIONS_H 1

/*********************************** Initialization **************************************/
bool getGpuInfo(int nGpu, int info, int deviceNumber);
void initNonlinearGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpu);
void allocateNonlinearGpu(double *vel2Dtw2, int iGpu);
void deallocateNonlinearGpu(int iGpu);

/*********************************** Nonlinear FWD **************************************/
void propShotsFwdGpu(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu);
void propShotsFwdGpuWavefield(double *modelRegDtw, double *dataRegDts, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu);

/*********************************** Nonlinear ADJ **************************************/
/* Adjoint propagation -- Data recorded at fine scale */
void propShotsAdjGpu(double *modelRegDtw, double *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu);
void propShotsAdjGpuWavefield(double *modelRegDtw, double *dataRegDtw, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *wavefieldDts, int iGpu);

#endif

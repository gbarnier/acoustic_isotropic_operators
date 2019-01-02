#ifndef BORN_SHOTS_GPU_FUNCTIONS_H
#define BORN_SHOTS_GPU_FUNCTIONS_H 1

/* Parameter settings */
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo);
void initBornGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpu);
void allocateBornShotsGpu(double *vel2Dtw2, double *refelctivityScale, int iGpu);
void deallocateBornShotsGpu(int iGpu);

/************************************** Born FWD ****************************************/
void BornShotsFwdGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *scatWavefield, int iGpu);
void BornShotsFwdGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu);

/************************************** Born ADJ ****************************************/
void BornShotsAdjGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu);
void BornShotsAdjGpuWavefield(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu);

#endif

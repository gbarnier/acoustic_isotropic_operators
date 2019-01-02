#ifndef BORN_EXT_SHOTS_GPU_FUNCTIONS_H
#define BORN_EXT_SHOTS_GPU_FUNCTIONS_H 1

/************************************** Initialization **********************************/
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo);
void initBornExtGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt, int nGpu, int iGpu);
void allocateBornExtShotsGpu(double *vel2Dtw2, double *refelctivityScale, int iGpu);
void deallocateBornExtShotsGpu(int iGpu);

/************************************** Born FWD ****************************************/
// Time-lags
void BornTimeShotsFwdGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *scatWavefield, int iGpu);
void BornTimeShotsFwdGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu);

// Subsurface offsets
void BornOffsetShotsFwdGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu);

/************************************** Born ADJ ****************************************/
// Time-lags
void BornTimeShotsAdjGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu);
void BornTimeShotsAdjGpuWavefield(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu);

// Subsurface offsets
void BornOffsetShotsAdjGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu);

#endif

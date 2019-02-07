#ifndef BORN_EXT_SHOTS_GPU_FUNCTIONS_H
#define BORN_EXT_SHOTS_GPU_FUNCTIONS_H 1

/************************************** Initialization **********************************/
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo);
void initBornExtGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nExt, int nGpu, int iGpu);
void allocateBornExtShotsGpu(float *vel2Dtw2, float *refelctivityScale, int iGpu);
void deallocateBornExtShotsGpu(int iGpu);

/************************************** Born FWD ****************************************/
// Time-lags
void BornTimeShotsFwdGpu(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *scatWavefield, int iGpu);
void BornTimeShotsFwdGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu);

// Subsurface offsets
void BornOffsetShotsFwdGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu);
void BornOffsetShotsFwdGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu);

/************************************** Born ADJ ****************************************/
// Time-lags
void BornTimeShotsAdjGpu(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu);
void BornTimeShotsAdjGpuWavefield(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu);

// Subsurface offsets
void BornOffsetShotsAdjGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu);
void BornOffsetShotsAdjGpuWavefield(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu);

#endif

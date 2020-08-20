#ifndef BORN_EXT_SHOTS_GPU_FUNCTIONS_H
#define BORN_EXT_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/************************************** Initialization **********************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initBornExtGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nExt, int nGpu, int iGpuId, int iGpuAlloc);
void allocateBornExtShotsGpu(float *vel2Dtw2, float *refelctivityScale, int iGpu, int iGpuId);
void deallocateBornExtShotsGpu(int iGpu, int iGpuId);

/************************************** Born FWD ****************************************/
// Time-lags
void BornTimeShotsFwdGpu(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *scatWavefield, int iGpu, int iGpuId);
void BornTimeShotsFwdGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId);

// Subsurface offsets
void BornOffsetShotsFwdGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId);
void BornOffsetShotsFwdGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId);

/************************************** Born ADJ ****************************************/
// Time-lags
void BornTimeShotsAdjGpu(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu, int iGpuId);
void BornTimeShotsAdjGpuWavefield(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu, int iGpuId);

// Subsurface offsets
void BornOffsetShotsAdjGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId);
void BornOffsetShotsAdjGpuWavefield(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu, int iGpuId);

#endif

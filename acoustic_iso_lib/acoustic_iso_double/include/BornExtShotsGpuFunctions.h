#ifndef BORN_EXT_SHOTS_GPU_FUNCTIONS_H
#define BORN_EXT_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/******************************* Initialization *******************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initBornExtGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt, int nGpu, int iGpuId, int iGpuAlloc);
void allocateBornExtShotsGpu(double *vel2Dtw2, double *refelctivityScale, int iGpu, int iGpuId);
void deallocateBornExtShotsGpu(int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Born forward **********************************/
/******************************************************************************/

/********************************** Normal ************************************/

// Time-lags
void BornTimeShotsFwdGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *scatWavefield, int sloth, int iGpu, int iGpuId);
void BornTimeShotsFwdGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);

// Subsurface offsets
void BornOffsetShotsFwdGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);
void BornOffsetShotsFwdGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);

/****************************** Free surface **********************************/
// Time-lags
void BornTimeShotsFwdFsGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *scatWavefield, int iGpu, int iGpuId);
void BornTimeShotsFwdFsGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);

// Subsurface offsets
void BornOffsetShotsFwdFsGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);
void BornOffsetShotsFwdFsGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Born adjoint **********************************/
/******************************************************************************/

/********************************** Normal ************************************/
// Time-lags
void BornTimeShotsAdjGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int sloth, int iGpu, int iGpuId);
void BornTimeShotsAdjGpuWavefield(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);

// Subsurface offsets
void BornOffsetShotsAdjGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);
void BornOffsetShotsAdjGpuWavefield(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);

/****************************** Free surface **********************************/
// Time-lags
void BornTimeShotsAdjFsGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);
void BornTimeShotsAdjFsGpuWavefield(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);

// Subsurface offsets
void BornOffsetShotsAdjFsGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);
void BornOffsetShotsAdjFsGpuWavefield(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);

#endif

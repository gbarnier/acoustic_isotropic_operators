#ifndef TOMO_EXT_SHOTS_GPU_FUNCTIONS_H
#define TOMO_EXT_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/******************************* Initialization *******************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initTomoExtGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nExt, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc);
void allocateTomoExtShotsGpu(float *vel2Dtw2, float *reflectivityScale, int iGpu, int iGpuId);
void deallocateTomoExtShotsGpu(int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Tomo forward **********************************/
/******************************************************************************/

/********************************** Normal ************************************/
void tomoTimeShotsFwdGpu(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void tomoOffsetShotsFwdGpu(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

/********************************** Adjoint ************************************/
void tomoTimeShotsFwdFsGpu(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void tomoOffsetShotsFwdFsGpu(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

/******************************************************************************/
/****************************** Tomo adjoint **********************************/
/******************************************************************************/

/********************************** Normal ************************************/
void tomoTimeShotsAdjGpu(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void tomoOffsetShotsAdjGpu(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

/********************************** Adjoint ************************************/
void tomoTimeShotsAdjFsGpu(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void tomoOffsetShotsAdjFsGpu(float *model, float *dataRegDts, float *extReflectivity, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

#endif

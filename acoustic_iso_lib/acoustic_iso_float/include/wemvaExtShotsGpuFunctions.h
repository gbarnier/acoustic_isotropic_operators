#ifndef WEMVA_EXT_SHOTS_GPU_FUNCTIONS_H
#define WEMVA_EXT_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/************************************** Initialization **********************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initWemvaExtGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nExt, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc);
void allocateWemvaExtShotsGpu(float *vel2Dtw2, float *reflectivityScale, int iGpu, int iGpuId);
void deallocateWemvaExtShotsGpu(int iGpu, int iGpuId);

/******************************** Wemva FWD ***********************************/
void wemvaTimeShotsFwdGpu(float *model, float *wemvaExtImage, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, float *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, float *wemvaSrcWavefieldDt2, float *wemvaSecWavefield1, float *wemvaSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void wemvaOffsetShotsFwdGpu(float *model, float *wemvaExtImage, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, float *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, float *wemvaSrcWavefieldDt2, float *wemvaSecWavefield1, float *wemvaSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

/******************************** Wemva ADJ ***********************************/
void wemvaTimeShotsAdjGpu(float *model, float *wemvaExtImage, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, float *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, float *wemvaSrcWavefieldDt2, float *wemvaSecWavefield1, float *wemvaSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void wemvaOffsetShotsAdjGpu(float *model, float *wemvaExtImage, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, float *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, float *wemvaSrcWavefieldDt2, float *wemvaSecWavefield1, float *wemvaSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

#endif

#ifndef WEMVA_EXT_SHOTS_GPU_FUNCTIONS_H
#define WEMVA_EXT_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/************************************** Initialization **********************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initWemvaExtGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc);
void allocateWemvaExtShotsGpu(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId);
void deallocateWemvaExtShotsGpu(int iGpu, int iGpuId);

/******************************** Wemva FWD ***********************************/
void wemvaTimeShotsFwdGpu(double *model, double *wemvaExtImage, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, double *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, double *wemvaSrcWavefieldDt2, double *wemvaSecWavefield1, double *wemvaSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void wemvaOffsetShotsFwdGpu(double *model, double *wemvaExtImage, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, double *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, double *wemvaSrcWavefieldDt2, double *wemvaSecWavefield1, double *wemvaSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

/******************************** Wemva ADJ ***********************************/
void wemvaTimeShotsAdjGpu(double *model, double *wemvaExtImage, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, double *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, double *wemvaSrcWavefieldDt2, double *wemvaSecWavefield1, double *wemvaSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void wemvaOffsetShotsAdjGpu(double *model, double *wemvaExtImage, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, double *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, double *wemvaSrcWavefieldDt2, double *wemvaSecWavefield1, double *wemvaSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

#endif

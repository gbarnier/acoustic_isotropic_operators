#ifndef TOMO_EXT_SHOTS_GPU_FUNCTIONS_H
#define TOMO_EXT_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/******************************* Initialization *******************************/
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initTomoExtGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt, int leg1, int leg2, int nGpu, int iGpuId, int iGpuAlloc);
void allocateTomoExtShotsGpu(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId);
void deallocateTomoExtShotsGpu(int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Tomo forward **********************************/
/******************************************************************************/

/********************************** Normal ************************************/
void tomoTimeShotsFwdGpu(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int sloth, int iGpu, int iGpuId, int saveWavefield);
void tomoOffsetShotsFwdGpu(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

/********************************** Adjoint ************************************/
void tomoTimeShotsFwdFsGpu(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void tomoOffsetShotsFwdFsGpu(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

/******************************************************************************/
/****************************** Tomo adjoint **********************************/
/******************************************************************************/

/********************************** Normal ************************************/
void tomoTimeShotsAdjGpu(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int sloth, int iGpu, int iGpuId, int saveWavefield);
void tomoOffsetShotsAdjGpu(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

/********************************** Adjoint ************************************/
void tomoTimeShotsAdjFsGpu(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);
void tomoOffsetShotsAdjFsGpu(double *model, double *dataRegDts, double *extReflectivity, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int iGpuId, int saveWavefield);

#endif

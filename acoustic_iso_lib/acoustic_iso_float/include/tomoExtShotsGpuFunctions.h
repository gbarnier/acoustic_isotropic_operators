#ifndef TOMO_EXT_SHOTS_GPU_FUNCTIONS_H
#define TOMO_EXT_SHOTS_GPU_FUNCTIONS_H 1

/************************************** Initialization **********************************/
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo);
void initTomoExtGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nExt, int leg1, int leg2, int nGpu, int iGpu);
void allocateTomoExtShotsGpu(float *vel2Dtw2, float *reflectivityScale, float *extReflectivity, int iGpu);
void deallocateTomoExtShotsGpu(int iGpu);

/************************************** Tomo FWD ****************************************/
void tomoTimeShotsFwdGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int saveWavefield);
void tomoOffsetShotsFwdGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int saveWavefield);

/************************************** Tomo ADJ ****************************************/
void tomoTimeShotsAdjGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int saveWavefield);
void tomoOffsetShotsAdjGpu(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *tomoSrcWavefieldDt2, float *tomoSecWavefield1, float *tomoSecWavefield2, int iGpu, int saveWavefield);

#endif

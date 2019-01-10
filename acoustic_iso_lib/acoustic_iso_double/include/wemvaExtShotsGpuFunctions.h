#ifndef WEMVA_EXT_SHOTS_GPU_FUNCTIONS_H
#define WEMVA_EXT_SHOTS_GPU_FUNCTIONS_H 1

/************************************** Initialization **********************************/
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo);
void initWemvaExtGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt, int leg1, int leg2, int nGpu, int iGpu);
void allocateWemvaExtShotsGpu(double *vel2Dtw2, double *reflectivityScale, int iGpu);
void deallocateWemvaExtShotsGpu(int iGpu);

/************************************** Tomo FWD ****************************************/
void wemvaExtShotsFwdGpu(double *model, double *wemvaExtImage, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, double *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, double *wemvaSrcWavefieldDt2, double *wemvaSecWavefield1, double *wemvaSecWavefield2, int iGpu, int saveWavefield, std::string extension);

/************************************** Tomo ADJ ****************************************/
void wemvaExtShotsAdjGpu(double *model, double *wemvaExtImage, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, double *wemvaDataRegDts, int *receiversPositionReg, int nReceiversReg, double *wemvaSrcWavefieldDt2, double *wemvaSecWavefield1, double *wemvaSecWavefield2, int iGpu, int saveWavefield, std::string extension);

#endif

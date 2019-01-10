#ifndef TOMO_EXT_SHOTS_GPU_FUNCTIONS_H
#define TOMO_EXT_SHOTS_GPU_FUNCTIONS_H 1

/************************************** Initialization **********************************/
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo);
void initTomoExtGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nExt, int leg1, int leg2, int nGpu, int iGpu);
void allocateTomoExtShotsGpu(double *vel2Dtw2, double *reflectivityScale, double *extReflectivity, int iGpu);
void deallocateTomoExtShotsGpu(int iGpu);

/************************************** Tomo FWD ****************************************/
void tomoExtShotsFwdGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int saveWavefield, std::string extension);

/************************************** Tomo ADJ ****************************************/
void tomoExtShotsAdjGpu(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *tomoSrcWavefieldDt2, double *tomoSecWavefield1, double *tomoSecWavefield2, int iGpu, int saveWavefield, std::string extension);

#endif

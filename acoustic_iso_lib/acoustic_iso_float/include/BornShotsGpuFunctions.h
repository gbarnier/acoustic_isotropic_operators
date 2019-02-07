#ifndef BORN_SHOTS_GPU_FUNCTIONS_H
#define BORN_SHOTS_GPU_FUNCTIONS_H 1

/* Parameter settings */
bool getGpuInfo(int nGpu, int info, int deviceNumberInfo);
void initBornGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpu);
void allocateBornShotsGpu(float *vel2Dtw2, float *refelctivityScale, int iGpu);
void deallocateBornShotsGpu(int iGpu);

/************************************** Born FWD ****************************************/
void BornShotsFwdGpu(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *scatWavefield, int iGpu);
void BornShotsFwdGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu);

/************************************** Born ADJ ****************************************/
void BornShotsAdjGpu(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu);
void BornShotsAdjGpuWavefield(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu);

#endif

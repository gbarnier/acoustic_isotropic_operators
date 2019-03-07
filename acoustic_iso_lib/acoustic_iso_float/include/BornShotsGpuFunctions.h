#ifndef BORN_SHOTS_GPU_FUNCTIONS_H
#define BORN_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/* Parameter settings */
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initBornGpu(float dz, float dx, int nz, int nx, int nts, float dts, int sub, int minPad, int blockSize, float alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateBornShotsGpu(float *vel2Dtw2, float *reflectivityScale, int iGpu, int iGpuId);
void deallocateBornShotsGpu(int iGpu, int iGpuId);

/************************************** Born FWD ****************************************/
void BornShotsFwdGpu(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *scatWavefield, int iGpu, int iGpuId);
void BornShotsFwdGpuWavefield(float *model, float *dataRegDts, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefieldDts, float *scatWavefieldDts, int iGpu, int iGpuId);

/************************************** Born ADJ ****************************************/
void BornShotsAdjGpu(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu, int iGpuId);
void BornShotsAdjGpuWavefield(float *model, float *dataRegDtw, float *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, float *srcWavefield, float *recWavefield, int iGpu, int iGpuId);

#endif

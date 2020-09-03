#ifndef BORN_SHOTS_GPU_FUNCTIONS_H
#define BORN_SHOTS_GPU_FUNCTIONS_H 1
#include <vector>

/* Parameter settings */
bool getGpuInfo(std::vector<int> gpuList, int info, int deviceNumberInfo);
void initBornGpu(double dz, double dx, int nz, int nx, int nts, double dts, int sub, int minPad, int blockSize, double alphaCos, int nGpu, int iGpuId, int iGpuAlloc);
void allocateBornShotsGpu(double *vel2Dtw2, double *reflectivityScale, int iGpu, int iGpuId);
void deallocateBornShotsGpu(int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Born forward **********************************/
/******************************************************************************/

/********************************** Normal ************************************/
void BornShotsFwdGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *scatWavefield, int iGpu, int iGpuId);
void BornShotsFwdGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);

/****************************** Free surface **********************************/
void BornShotsFwdFsGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *scatWavefield, int iGpu, int iGpuId);
void BornShotsFwdFsGpuWavefield(double *model, double *dataRegDts, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefieldDts, double *scatWavefieldDts, int iGpu, int iGpuId);

/******************************************************************************/
/****************************** Born adjoint **********************************/
/******************************************************************************/

/********************************** Normal ************************************/
void BornShotsAdjGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);
void BornShotsAdjGpuWavefield(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);

/******************************** Free surface ********************************/
void BornShotsAdjFsGpu(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);
void BornShotsAdjFsGpuWavefield(double *model, double *dataRegDtw, double *sourcesSignals, int *sourcesPositionReg, int nSourcesReg, int *receiversPositionReg, int nReceiversReg, double *srcWavefield, double *recWavefield, int iGpu, int iGpuId);

#endif

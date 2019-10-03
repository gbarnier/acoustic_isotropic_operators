#ifndef WAVE_EQUATION_ACOUSTIC_GPU_FUNCTIONS_H
#define WAVE_EQUATION_ACOUSTIC_GPU_FUNCTIONS_H 1

/*********************************** Initialization **************************************/
//bool getGpuInfo(int nGpu, int info, int deviceNumber);
void initWaveEquationAcousticGpu(float dz, float dx, int nz, int nx, int nts, float dts, int minPad, int blockSize, int nGpu, int iGpuId, int iGpuAlloc);
void allocateWaveEquationAcousticGpu(float *slsqDt2, float *cosDamp, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample);
void deallocateWaveEquationAcousticGpu();
void waveEquationAcousticFwdGpu(float *model,float *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample);
void waveEquationAcousticAdjGpu(float *model,float *data, int iGpu, int iGpuId, int firstTimeSample, int lastTimeSample);
bool getGpuInfo(int nGpu, int info, int deviceNumber);
float getTotalGlobalMem(int nGpu, int info, int deviceNumber); //return GB of total global mem

#endif


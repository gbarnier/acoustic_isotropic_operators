#ifndef WEMVA_EXT_GPU_H
#define WEMVA_EXT_GPU_H 1

#include <string>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"
#include "deviceGpu.h"
#include "fdParam.h"
#include "operator.h"
#include "interpTimeLinTbb.h"
#include "secondTimeDerivative.h"
#include "wemvaExtShotsGpuFunctions.h"

using namespace SEP;

class wemvaExtGpu : public Operator<SEP::float2DReg, SEP::float3DReg> {

	private:

		std::shared_ptr<fdParam> _fdParam;
		std::shared_ptr<deviceGpu> _sources, _receivers;
		int *_sourcesPositionReg, *_receiversPositionReg;
		int _nSourcesReg, _nReceiversReg;
		int _nts;
		int _saveWavefield;
		int _iGpu, _nGpu;
		int _leg1, _leg2;
		std::shared_ptr<interpTimeLinTbb> _timeInterp;
		std::shared_ptr<secondTimeDerivative> _secTimeDer;
		std::shared_ptr<float3DReg> _srcWavefield, _secWavefield1, _secWavefield2;
		std::shared_ptr<SEP::float2DReg> _sourcesSignals, _sourcesSignalsRegDts, _sourcesSignalsRegDtsDt2, _sourcesSignalsRegDtwDt2, _sourcesSignalsRegDtw;
		std::shared_ptr<SEP::float2DReg> _receiversSignals, _receiversSignalsRegDts;

	public:

		/* Overloaded constructors */
		wemvaExtGpu(std::shared_ptr<SEP::float2DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu);

		/* Acquisition setup */
		void setSources(std::shared_ptr<deviceGpu> sources, std::shared_ptr<SEP::float2DReg> sourcesSignals);
		void setReceivers(std::shared_ptr<deviceGpu> receivers, std::shared_ptr<SEP::float2DReg> receiversSignals);
		void setAcquisition(std::shared_ptr<deviceGpu> sources, std::shared_ptr<SEP::float2DReg> sourcesSignals, std::shared_ptr<deviceGpu> receivers, std::shared_ptr<SEP::float2DReg> receiversSignals, const std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float3DReg> data);

		/* Wavefields */
		void setAllWavefields(int wavefieldFlag);

		/* Other functions */
		void scaleSeismicSource(const std::shared_ptr<deviceGpu> seismicSource, std::shared_ptr<SEP::float2DReg> signal, const std::shared_ptr<fdParam> parObj);
		void setGpuNumber(int iGpu){_iGpu = iGpu;}
		std::shared_ptr<float3DReg> setWavefield(int wavefieldFlag); // If flag=1, allocates a wavefield (with the correct size) and returns it. If flag=0, return a dummy wavefield of size 1x1x1

		/* QC */
		bool checkParfileConsistency(const std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float3DReg> data) const;

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float3DReg> data) const;

		/* Destructor */
		~wemvaExtGpu(){};

		/* Accessors */
		std::shared_ptr<float3DReg> getSrcWavefield() { return _srcWavefield; }
		std::shared_ptr<float3DReg> getSecWavefield1() { return _secWavefield1; }
		std::shared_ptr<float3DReg> getSecWavefield2() { return _secWavefield2; } // When adj=1, this is the receiver wavefield
		std::shared_ptr<fdParam> getFdParam(){ return _fdParam; }

};

#endif

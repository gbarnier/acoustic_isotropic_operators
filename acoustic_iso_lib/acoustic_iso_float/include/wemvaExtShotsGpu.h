#ifndef WEMVA_EXT_SHOTS_GPU_H
#define WEMVA_EXT_SHOTS_GPU_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"
#include "deviceGpu.h"
#include "fdParam.h"
#include "operator.h"
#include "interpTimeLinTbb.h"
#include "wemvaExtShotsGpuFunctions.h"

using namespace SEP;

class wemvaExtShotsGpu : public Operator<SEP::float2DReg, SEP::float3DReg> {

	private:
		int _nShot, _nGpu, _iGpuAlloc;
		int _saveWavefield, _wavefieldShotNumber, _info, _deviceNumberInfo;
		std::shared_ptr<SEP::float2DReg> _vel;
		std::vector<std::shared_ptr<SEP::float2DReg>> _sourcesSignalsVector, _receiversSignalsVector;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu>> _sourcesVector, _receiversVector;
		std::shared_ptr<SEP::float3DReg> _srcWavefield, _secWavefield1, _secWavefield2;
		std::shared_ptr<SEP::float3DReg> _reflectivityExt;
		std::vector<int> _gpuList;

	public:

		/* Overloaded constructors */
		wemvaExtShotsGpu(std::shared_ptr<SEP::float2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<SEP::float2DReg>> sourcesSignalsVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector, std::vector<std::shared_ptr<SEP::float2DReg>> receiversSignalsVector);

		/* Destructor */
		~wemvaExtShotsGpu(){};

		/* Create Gpu list */
		void createGpuIdList();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float3DReg> data) const;
		void forwardWavefield(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float3DReg> data);
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float3DReg> data) const;
		void adjointWavefield(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float3DReg> data);

		/* Accessor */
		std::shared_ptr<float3DReg> getSrcWavefield(){ return _srcWavefield; }
		std::shared_ptr<float3DReg> getSecWavefield1() { return _secWavefield1; }
		std::shared_ptr<float3DReg> getSecWavefield2() { return _secWavefield2; }

		/* Mutators */
		void setVel(std::shared_ptr<SEP::float2DReg> vel){ _vel = vel; }
		void setReceiversSignalsVector(std::vector<std::shared_ptr<SEP::float2DReg>> receiversSignalsVector){ _receiversSignalsVector = receiversSignalsVector; }

};

#endif

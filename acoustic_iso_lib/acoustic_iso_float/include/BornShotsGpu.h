#ifndef BORN_SHOTS_GPU_H
#define BORN_SHOTS_GPU_H 1

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
#include "BornShotsGpuFunctions.h"

using namespace SEP;

class BornShotsGpu : public Operator<SEP::float2DReg, SEP::float3DReg> {

	private:
		int _nShot, _nGpu;
		int _saveWavefield, _wavefieldShotNumber, _info, _deviceNumberInfo;
		std::shared_ptr<SEP::float2DReg> _vel;
		std::vector<std::shared_ptr<SEP::float2DReg>> _sourcesSignalsVector;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu>> _sourcesVector, _receiversVector;
		std::shared_ptr<SEP::float3DReg> _srcWavefield, _secWavefield;

	public:

		/* Overloaded constructors */
		BornShotsGpu(std::shared_ptr<SEP::float2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<SEP::float2DReg>> sourcesSignalsVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector);

		/* Destructor */
		~BornShotsGpu(){};

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float3DReg> data) const;
		void forwardWavefield(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float3DReg> data);
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float3DReg> data) const;
		void adjointWavefield(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float3DReg> data);

		/* Accessor */
		std::shared_ptr<float3DReg> getSrcWavefield(){ return _srcWavefield; }
		std::shared_ptr<float3DReg> getSecWavefield(){ return _secWavefield; }

		/* Mutators */
		void setVel(std::shared_ptr<SEP::float2DReg> vel){ _vel = vel; }

};

#endif

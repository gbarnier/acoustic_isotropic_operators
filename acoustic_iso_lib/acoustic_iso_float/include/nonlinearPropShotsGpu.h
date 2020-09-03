#ifndef NL_PROP_SHOTS_GPU_H
#define NL_PROP_SHOTS_GPU_H 1

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

using namespace SEP;

class nonlinearPropShotsGpu : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:
		int _nShot, _nGpu, _info, _deviceNumberInfo, _iGpuAlloc;
		int _saveWavefield, _wavefieldShotNumber;
		std::shared_ptr<SEP::float2DReg> _vel;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu>> _sourcesVector, _receiversVector;
		std::shared_ptr<SEP::float3DReg> _wavefield;
		std::vector<int> _gpuList;

	public:

		/* Overloaded constructors */
		nonlinearPropShotsGpu(std::shared_ptr<SEP::float2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector);

		/* Destructor */
		~nonlinearPropShotsGpu(){};

		/* Create Gpu list */
		void createGpuIdList();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
		void forwardWavefield(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data);
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;
		void adjointWavefield(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data);

		/* Accessor */
		std::shared_ptr<SEP::float3DReg> getWavefield(){
			std::cout << "Returning wavefield" << std::endl;
			return _wavefield;
		}

		/* Mutator */
		void setVel(std::shared_ptr<SEP::float2DReg> vel){_vel = vel;}
};

#endif

#ifndef NL_PROP_SHOTS_GPU_H
#define NL_PROP_SHOTS_GPU_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <vector>
#include "double2DReg.h"
#include "double3DReg.h"
#include "ioModes.h"
#include "deviceGpu.h"
#include "fdParam.h"
#include "operator.h"

using namespace SEP;

class nonlinearPropShotsGpu : public Operator<SEP::double3DReg, SEP::double3DReg> {

	private:
		int _nShot, _nGpu, _info, _deviceNumberInfo, _iGpuAlloc;
		int _saveWavefield, _wavefieldShotNumber;
		std::shared_ptr<SEP::double2DReg> _vel;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu>> _sourcesVector, _receiversVector;
		std::shared_ptr<SEP::double3DReg> _wavefield;
		std::vector<int> _gpuList;

	public:

		/* Overloaded constructors */
		nonlinearPropShotsGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector);

		/* Destructor */
		~nonlinearPropShotsGpu(){};

		/* Create Gpu list */
		void createGpuIdList();

		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const;
		void forwardWavefield(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data);
		void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const;
		void adjointWavefield(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data);

		/* Accessor */
		std::shared_ptr<SEP::double3DReg> getWavefield(){
			return _wavefield;
		}

		/* Mutator */
		void setVel(std::shared_ptr<SEP::double2DReg> vel){_vel = vel;}
};

#endif

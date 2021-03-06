#ifndef NL_PROP_GPU_H
#define NL_PROP_GPU_H 1

#include <string>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "double2DReg.h"
#include "double3DReg.h"
#include "ioModes.h"
#include "deviceGpu.h"
#include "fdParam.h"
#include "seismicOperator2D.h"
#include "interpTimeLinTbb.h"
#include "nonlinearShotsGpuFunctions.h"

using namespace SEP;

class nonlinearPropGpu : public seismicOperator2D<SEP::double2DReg, SEP::double2DReg> {

	private:

		std::shared_ptr<double3DReg> _wavefield;

	public:

		/* Overloaded constructors */
		nonlinearPropGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		/* Mutators */
		void setAllWavefields(int wavefieldFlag);

		/* QC */
		bool checkParfileConsistency(const std::shared_ptr<SEP::double2DReg> model, const std::shared_ptr<SEP::double2DReg> data) const;

		// FWD / ADJ
		void forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const;

		// Destructor
		~nonlinearPropGpu(){};

		// Accessors
		std::shared_ptr<double3DReg> getWavefield() { return _wavefield; }

		std::shared_ptr<double2DReg> _dataDtw;

};

#endif

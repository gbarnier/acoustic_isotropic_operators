#ifndef TOMO_EXT_GPU_H
#define TOMO_EXT_GPU_H 1

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
#include "secondTimeDerivative.h"
#include "tomoExtShotsGpuFunctions.h"

using namespace SEP;

class tomoExtGpu : public seismicOperator2D<SEP::double2DReg, SEP::double2DReg> {

	private:

		int _sloth;
		int _leg1, _leg2;
		std::shared_ptr<double3DReg> _srcWavefield, _secWavefield1, _secWavefield2;
		std::shared_ptr<double3DReg> _reflectivityExt;

	public:

		/* Overloaded constructors */
		tomoExtGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, std::shared_ptr<double3DReg> reflectivityExt, int nGpu, int iGpu, int iGpuId, int iGpuAlloc);

		/* Mutators */
		void setReflectivityExt(std::shared_ptr<double3DReg> reflectivityExt){ _reflectivityExt=reflectivityExt;}
		bool checkParfileConsistency(const std::shared_ptr<SEP::double2DReg> model, const std::shared_ptr<SEP::double2DReg> data) const;
		void setAllWavefields(int wavefieldFlag); // Allocates all wavefields assocaited with a seismic operator

		/* FWD - ADJ */
		void forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const;

		/* Destructor */
		~tomoExtGpu(){};

		/* Accessors */
		std::shared_ptr<double3DReg> getSrcWavefield() { return _srcWavefield; }
		std::shared_ptr<double3DReg> getSecWavefield1() { return _secWavefield1; }
		std::shared_ptr<double3DReg> getSecWavefield2() { return _secWavefield2; } // When adj, this is the receiver wavefield
		std::shared_ptr<double3DReg> getReflectivityExt() { return _reflectivityExt; }

};

#endif

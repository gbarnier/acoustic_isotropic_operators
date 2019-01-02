#ifndef BORN_EXT_SHOTS_GPU_H
#define BORN_EXT_SHOTS_GPU_H 1

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
#include "interpTimeLinTbb.h"
#include "BornExtShotsGpuFunctions.h"

using namespace SEP;

class BornExtShotsGpu : public Operator<SEP::double3DReg, SEP::double3DReg> {
	
	private:
		int _nShot, _nGpu;
		int _saveWavefield, _wavefieldShotNumber, _info, _deviceNumberInfo;
		std::shared_ptr<SEP::double2DReg> _vel;
		std::vector<std::shared_ptr<SEP::double2DReg>> _sourcesSignalsVector;
		std::shared_ptr<paramObj> _par;
		std::vector<std::shared_ptr<deviceGpu>> _sourcesVector, _receiversVector;
		std::shared_ptr<SEP::double3DReg> _srcWavefield, _secWavefield;
	
	public:

		/* Overloaded constructors */ 	
		BornExtShotsGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<SEP::double2DReg>> sourcesSignalsVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector);
		
		/* Destructor */
		~BornExtShotsGpu(){};
		
		/* FWD / ADJ */
		void forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const;	
		void forwardWavefield(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data);	
		void adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const; 	
		void adjointWavefield(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data); 	
								
		/* Accessor */
		std::shared_ptr<double3DReg> getSrcWavefield(){ return _srcWavefield; }
		std::shared_ptr<double3DReg> getSecWavefield(){ return _secWavefield; }		
			
};

#endif	

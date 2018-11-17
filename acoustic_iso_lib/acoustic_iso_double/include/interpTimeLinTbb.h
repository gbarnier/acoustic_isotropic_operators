#ifndef INTERPTTIMELIN_TBB_H
#define INTERPTTIMELIN_TBB_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "operator.h"
#include "double1DReg.h"
#include "double2DReg.h"


using namespace SEP;

class interpTimeLinTbb : public Operator<SEP::double2DReg, SEP::double2DReg>
{
	private:
	
		int _nts, _ntw;
		double _dts, _dtw, _scale;
		double _ots, _otw;
		int _sub;
		axis _timeAxisCoarse, _timeAxisFine;

	public:

		/* Overloaded constructor */ 	
		interpTimeLinTbb(int nts, double dts, double ots, int sub);
	
		/* Destructor */ 	
		~interpTimeLinTbb(){};
	
  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const;
		void forward2(std::shared_ptr<double2DReg> toto);		
};

#endif

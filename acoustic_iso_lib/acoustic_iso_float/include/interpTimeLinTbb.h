#ifndef INTERPTTIMELIN_TBB_H
#define INTERPTTIMELIN_TBB_H 1

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "operator.h"
#include "float1DReg.h"
#include "float2DReg.h"


using namespace SEP;

class interpTimeLinTbb : public Operator<SEP::float2DReg, SEP::float2DReg>
{
	private:

		int _nts, _ntw;
		float _dts, _dtw, _scale;
		float _ots, _otw;
		int _sub;
		axis _timeAxisCoarse, _timeAxisFine;

	public:

		/* Overloaded constructor */
		interpTimeLinTbb(int nts, float dts, float ots, int sub);

		/* Destructor */
		~interpTimeLinTbb(){};

  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const;
		void forward2(std::shared_ptr<float2DReg> toto);
};

#endif

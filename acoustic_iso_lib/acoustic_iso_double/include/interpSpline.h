#ifndef INTERP_SPLINE_H
#define INTERP_SPLINE_H 1

#include "operator.h"
#include "int1DReg.h"
#include "double1DReg.h"

using namespace SEP;

class interpSpline1d : public Operator<SEP::double1DReg, SEP::double1DReg>
{
	private:

		int _nModel, _nData, _nInterval;
		int _n1;
		double _o1, _d1;
		std::shared_ptr<int1DReg> _modelGridPos, _dataGridPos, _interpFilter;
		axis _gridAxis;

	public:

		/* Overloaded constructor */
		interpSplineLinear1d(std::shared_ptr<int1DReg> _modelGridPos, std::shared_ptr<int1DReg> _dataGridPos);

		/* Destructor */
		~interpSplineLinear1d(){};

  		/* Forward / Adjoint */
  		virtual void forward(const bool add, const std::shared_ptr<double1DReg> model, std::shared_ptr<double1DReg> data) const;
 		virtual void adjoint(const bool add, std::shared_ptr<double1DReg> model, const std::shared_ptr<double1DReg> data) const;

};

#endif

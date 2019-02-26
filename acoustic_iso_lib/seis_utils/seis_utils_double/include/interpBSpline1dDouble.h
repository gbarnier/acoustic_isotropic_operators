#ifndef INTERP_BSPLINE_1D_DOUBLE_H
#define INTERP_BSPLINE_1D_DOUBLE_H 1

#include "operator.h"
#include "double1DReg.h"
#include <vector>

using namespace SEP;

class interpBSpline1dDouble : public Operator<SEP::double1DReg, SEP::double1DReg> {

	private:

		int _order, _nk, _nkSimple, _nModel, _nData, _scaling, _nParam, _fat;
		double _ok, _dk, _fk, _tolerance;
		axis _kAxis, _xDataAxis;
		std::shared_ptr<double1DReg> _xModel, _xData, _knots, _paramVector, _scaleVector, _curveX, _curveY;

	public:

		/* Overloaded constructors */
		interpBSpline1dDouble(int order, std::shared_ptr<double1DReg> xModel, axis xDataAxis, int nParam, int scaling, double tolerance, int fat);

		/* Spline interpolation function */
		double bspline1d(int iControl, double uValue, int order, double dk, int nControlPoints, std::shared_ptr<double1DReg> knots) const;

        /* Compute Knot vector */
        std::shared_ptr<double1DReg> buildKnotVector();

		/* Compute vector containing optimal parameters for each data points */
		std::shared_ptr<double1DReg> computeParamVector();

		/* Compute scale vector */
		std::shared_ptr<double1DReg> computeScaleVector(int scaling);

		/* Compute entire curve */
		std::shared_ptr<double1DReg> computeCurve(int nSample, std::shared_ptr<double1DReg> controlPoints);

		/* Forward / Adjoint */
		void forward(const bool add, const std::shared_ptr<double1DReg> model, std::shared_ptr<double1DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double1DReg> model, const std::shared_ptr<double1DReg> data) const;
};

#endif

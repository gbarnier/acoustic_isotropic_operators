#ifndef INTERP_BSPLINE_1D_H
#define INTERP_BSPLINE_1D_H 1

#include "operator.h"
#include "float1DReg.h"
#include <vector>

using namespace SEP;

class interpBSpline1d : public Operator<SEP::float1DReg, SEP::float1DReg> {

	private:

		int _zOrder, _nkz, _nkzSimple, _nModel, _nData, _nzParamVector, _scaling, _nzModel, _nzData, _fat;
		float _okz, _dkz, _dkz2, _dkz3, _fkz, _zTolerance;
		axis _kzAxis, _zDataAxis;
		std::shared_ptr<float1DReg> _zKnots, _zParamVector, _zMeshVector, _zModel, _zData, _scaleVector, _interpCurve;

	public:

		// Overloaded constructors
		interpBSpline1d(int zOrder, std::shared_ptr<float1DReg> zModel, axis zDataAxis, int nzParamVector, int scaling, float zTolerance, int fat);

		// Knot vector
        void buildKnotVector();

		// Compute vector containing optimal parameters for each data points
		std::shared_ptr<float1DReg> computeParamVectorZ();

		// Scaling vector
		void computeScaleVector();

		// Accessors
		std::shared_ptr<float1DReg> getZMesh(){ return _zModel;}

		// Forward / Adjoint
		void forward(const bool add, const std::shared_ptr<float1DReg> model, std::shared_ptr<float1DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float1DReg> model, const std::shared_ptr<float1DReg> data) const;

};

#endif

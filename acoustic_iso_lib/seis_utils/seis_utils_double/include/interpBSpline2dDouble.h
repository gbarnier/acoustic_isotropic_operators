#ifndef INTERP_BSPLINE_2D_DOUBLE_H
#define INTERP_BSPLINE_2D_DOUBLE_H 1

#include "operator.h"
#include "double1DReg.h"
#include "double2DReg.h"
#include <vector>

using namespace SEP;

class interpBSpline2dDouble : public Operator<SEP::double2DReg, SEP::double2DReg> {

	private:

		int _zOrder, _xOrder, _nkz, _nkzSimple, _nkx, _nkxSimple, _nModel, _nData, _nzParamVector, _nxParamVector, _scaling, _nzModel, _nxModel, _nzData, _nxData, _fat;
		double _okz, _dkz, _fkz, _okx, _dkx, _fkx, _zTolerance, _xTolerance;
		axis _kzAxis, _kxAxis, _zDataAxis, _xDataAxis;
		std::shared_ptr<double1DReg> _zControlPoints, _xControlPoints, _zKnots, _xKnots, _zParamVector, _xParamVector;
		std::shared_ptr<double2DReg> _zModel, _xModel, _zData, _xData, _scaleVector, _interpSurfaceZ, _interpSurfaceX, _interpSurfaceVel;

	public:

		// Overloaded constructors
		interpBSpline2dDouble(int zOrder, int xOrder, std::shared_ptr<double1DReg> _zControlPoints, std::shared_ptr<double1DReg> _xControlPoints, axis zDataAxis, axis xDataAxis, int nzParamVector, int nxParamVector, int scaling, double zTolerance, double xTolerance, int fat);

		interpBSpline2dDouble(int zOrder, int xOrder, std::shared_ptr<double1DReg> _zControlPoints, std::shared_ptr<double1DReg> _xControlPoints, axis zDataAxis, axis xDataAxis, std::shared_ptr<double1DReg> zParamVector, std::shared_ptr<double1DReg> xParamVector, int scaling, double zTolerance, double xTolerance, int fat);

		// Spline interpolation function
		double bspline1d(int iControl, double uValue, int order, double dk, int nControlPoints, std::shared_ptr<double1DReg> knots);

		// Knot vector
        void buildKnotVectors2d();

		// Compute vector containing optimal parameters for each data points
		std::shared_ptr<double1DReg> computeParamVectorZ();
		std::shared_ptr<double1DReg> computeParamVectorX();

		// Scaling vector
		std::shared_ptr<double2DReg> computeScaleVector(int scalingOn);

		// Compute surface
		std::shared_ptr<double2DReg> computeSurface(int nu, int nv, std::shared_ptr<double2DReg> valModel);

		// Accessors
		std::shared_ptr<double2DReg> getInterpSurfaceZ(){ return _interpSurfaceZ; }
		std::shared_ptr<double2DReg> getInterpSurfaceX(){ return _interpSurfaceX; }
		std::shared_ptr<double2DReg> getZModel(){ return _zModel; }
		std::shared_ptr<double2DReg> getXModel(){ return _xModel; }
		std::shared_ptr<double1DReg> getZParamVector(){ return _zParamVector; }
		std::shared_ptr<double1DReg> getXParamVector(){ return _xParamVector; }
		std::shared_ptr<double1DReg> getZKnots(){ return _zKnots; }
		std::shared_ptr<double1DReg> getXKnots(){ return _xKnots; }
		std::shared_ptr<double2DReg> getScaleVector(){ return _scaleVector;}

		// Mutators
		void setParamVectors(std::shared_ptr<double1DReg> zParamVector, std::shared_ptr<double1DReg> xParamVector);

		// Forward / Adjoint
		void forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const;
		void forwardNoScale(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const;
		void adjointNoScale(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const;

};

#endif

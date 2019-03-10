#ifndef INTERP_BSPLINE_2D_H
#define INTERP_BSPLINE_2D_H 1

#include "operator.h"
#include "float1DReg.h"
#include "float2DReg.h"
#include <vector>

using namespace SEP;

class interpBSpline2d : public Operator<SEP::float2DReg, SEP::float2DReg> {

	private:

		int _zOrder, _xOrder, _nkz, _nkzSimple, _nkx, _nkxSimple, _nModel, _nData, _nzParamVector, _nxParamVector, _scaling, _nzModel, _nxModel, _nzData, _nxData, _fat;
		float _okz, _dkz, _dkz2, _dkz3, _fkz, _okx, _dkx, _dkx2, _dkx3, _fkx, _zTolerance, _xTolerance;
		axis _kzAxis, _kxAxis, _zDataAxis, _xDataAxis;
		std::shared_ptr<float1DReg> _zControlPoints, _xControlPoints, _zKnots, _xKnots, _zParamVector, _xParamVector, _zMeshVector, _xMeshVector, _zMeshDataVector, _xMeshDataVector;
		std::shared_ptr<float2DReg> _zModel, _xModel, _zData, _xData, _scaleVector, _interpSurfaceZ, _interpSurfaceX, _interpSurfaceVel;

	public:

		// Overloaded constructors
		interpBSpline2d(int zOrder, int xOrder, std::shared_ptr<float1DReg> _zControlPoints, std::shared_ptr<float1DReg> _xControlPoints, axis zDataAxis, axis xDataAxis, int nzParamVector, int nxParamVector, int scaling, float zTolerance, float xTolerance, int fat);

		interpBSpline2d(int zOrder, int xOrder, std::shared_ptr<float1DReg> _zControlPoints, std::shared_ptr<float1DReg> _xControlPoints, axis zDataAxis, axis xDataAxis, std::shared_ptr<float1DReg> zParamVector, std::shared_ptr<float1DReg> xParamVector, int scaling, float zTolerance, float xTolerance, int fat);

		// Spline interpolation function
		float bspline1d(int iControl, float uValue, int order, float dk, int nControlPoints, std::shared_ptr<float1DReg> knots);

		// Knot vector
        void buildKnotVectors2d();

		// Compute vector containing optimal parameters for each data points
		std::shared_ptr<float1DReg> computeParamVectorZ();
		std::shared_ptr<float1DReg> computeParamVectorX();

		// Scaling vector
		std::shared_ptr<float2DReg> computeScaleVector(int scalingOn);

		// Compute surface
		std::shared_ptr<float2DReg> computeSurface(int nu, int nv, std::shared_ptr<float2DReg> valModel);

		// Accessors
		std::shared_ptr<float2DReg> getInterpSurfaceZ(){ return _interpSurfaceZ; }
		std::shared_ptr<float2DReg> getInterpSurfaceX(){ return _interpSurfaceX; }
		std::shared_ptr<float2DReg> getZModel(){ return _zModel; }
		std::shared_ptr<float2DReg> getXModel(){ return _xModel; }
		std::shared_ptr<float1DReg> getZParamVector(){ return _zParamVector; }
		std::shared_ptr<float1DReg> getXParamVector(){ return _xParamVector; }
		std::shared_ptr<float1DReg> getZKnots(){ return _zKnots; }
		std::shared_ptr<float1DReg> getXKnots(){ return _xKnots; }
		std::shared_ptr<float2DReg> getScaleVector(){ return _scaleVector;}
		std::shared_ptr<float1DReg> getZMesh(){ return _zMeshVector;}
		std::shared_ptr<float1DReg> getXMesh(){ return _xMeshVector;}
		std::shared_ptr<float1DReg> getZMeshData();
		std::shared_ptr<float1DReg> getXMeshData();

		// Forward / Adjoint
		void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const;
		void forwardNoScale(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const;
		void adjointNoScale(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const;

};

#endif

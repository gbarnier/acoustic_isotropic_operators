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
		std::shared_ptr<float1DReg> _zControlPoints, _xControlPoints, _zKnots, _xKnots, _zParamVector, _xParamVector, _zMeshModelVector, _xMeshModelVector, _zMeshDataVector, _xMeshDataVector, _zModel, _xModel, _zData, _xData;
		std::shared_ptr<float2DReg> _scaleVector, _interpSurfaceZ, _interpSurfaceX, _interpSurfaceVel;

	public:

		// Overloaded constructors
		interpBSpline2d(int zOrder, int xOrder, std::shared_ptr<float1DReg> _zControlPoints, std::shared_ptr<float1DReg> _xControlPoints, axis zDataAxis, axis xDataAxis, int nzParamVector, int nxParamVector, int scaling, float zTolerance, float xTolerance, int fat);

		// Knot vector
        void buildKnotVectors2d();

		// Compute vector containing optimal parameters for each data points
		std::shared_ptr<float1DReg> computeParamVectorZ();
		std::shared_ptr<float1DReg> computeParamVectorX();

		// Scaling vector
		std::shared_ptr<float2DReg> computeScaleVector();

		// Accessors
		std::shared_ptr<float1DReg> getZParamVector(){ return _zParamVector; }
		std::shared_ptr<float1DReg> getXParamVector(){ return _xParamVector; }
		std::shared_ptr<float1DReg> getZKnots(){ return _zKnots; }
		std::shared_ptr<float1DReg> getXKnots(){ return _xKnots; }
		std::shared_ptr<float2DReg> getScaleVector(){ return _scaleVector;}
		std::shared_ptr<float1DReg> getZMeshModel();
		std::shared_ptr<float1DReg> getXMeshModel();
		std::shared_ptr<float1DReg> getZMeshData();
		std::shared_ptr<float1DReg> getXMeshData();
		std::shared_ptr<float1DReg> getZControlPoints(){return _zControlPoints;}
		std::shared_ptr<float1DReg> getXControlPoints(){return _xControlPoints;}

		// Forward / Adjoint
		void forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const;

};

#endif

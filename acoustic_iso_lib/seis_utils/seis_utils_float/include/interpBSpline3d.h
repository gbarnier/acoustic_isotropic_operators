#ifndef INTERP_BSPLINE_3D_H
#define INTERP_BSPLINE_3D_H 1

#include "operator.h"
#include "float1DReg.h"
#include "float3DReg.h"
#include <vector>

using namespace SEP;

class interpBSpline3d : public Operator<SEP::float3DReg, SEP::float3DReg> {

	private:

		int _zOrder, _xOrder, _yOrder, _nkz, _nkzSimple, _nkx, _nkxSimple, _nky, _nkySimple, _nModel, _nData, _nzParamVector, _nxParamVector, _nyParamVector, _scaling, _nzModel, _nxModel, _nyModel, _nzData, _nxData, _nyData, _zFat, _xFat, _yFat;
		float _okz, _dkz, _dkz2, _dkz3, _fkz, _okx, _dkx, _dkx2, _dkx3, _fkx, _oky, _dky, _dky2, _dky3, _fky, _zTolerance, _xTolerance, _yTolerance;
		axis _kzAxis, _kxAxis, _kyAxis, _zDataAxis, _xDataAxis, _yDataAxis;
		std::shared_ptr<float1DReg> _zControlPoints, _xControlPoints, _yControlPoints, _zKnots, _xKnots, _yKnots, _zParamVector, _xParamVector, _yParamVector, _zMeshModelVector, _xMeshModelVector, _yMeshModelVector, _zMeshDataVector, _xMeshDataVector, _yMeshDataVector, _zData, _xData, _yData;
		std::shared_ptr<float3DReg> _scaleVector;

	public:

		// Overloaded constructors
		interpBSpline3d(int zOrder, int xOrder, int yOrder, std::shared_ptr<float1DReg> zControlPoints, std::shared_ptr<float1DReg> xControlPoints, std::shared_ptr<float1DReg> yControlPoints, axis zDataAxis, axis xDataAxis, axis yDataAxis, int nzParamVector, int nxParamVector, int nyParamVector, int scaling, float zTolerance, float xTolerance, float yTolerance, int zFat, int xFat, int yFat);

		// Knot vector
        void buildKnotVectors3d();

		// Compute vector containing optimal parameters for each data points
		std::shared_ptr<float1DReg> computeParamVectorZ();
		std::shared_ptr<float1DReg> computeParamVectorX();
		std::shared_ptr<float1DReg> computeParamVectorY();

		// Scaling vector
		std::shared_ptr<float3DReg> computeScaleVector();

		// // Accessors
		std::shared_ptr<float1DReg> getZParamVector(){ return _zParamVector; }
		std::shared_ptr<float1DReg> getXParamVector(){ return _xParamVector; }
		std::shared_ptr<float1DReg> getYParamVector(){ return _yParamVector; }
		std::shared_ptr<float1DReg> getZKnots(){ return _zKnots; }
		std::shared_ptr<float1DReg> getXKnots(){ return _xKnots; }
		std::shared_ptr<float1DReg> getYKnots(){ return _yKnots; }
		std::shared_ptr<float3DReg> getScaleVector(){ return _scaleVector;}
		std::shared_ptr<float1DReg> getZMeshModel();
		std::shared_ptr<float1DReg> getXMeshModel();
		std::shared_ptr<float1DReg> getYMeshModel();
		std::shared_ptr<float1DReg> getZMeshData();
		std::shared_ptr<float1DReg> getXMeshData();
		std::shared_ptr<float1DReg> getYMeshData();
		std::shared_ptr<float1DReg> getZControlPoints(){return _zControlPoints;}
		std::shared_ptr<float1DReg> getXControlPoints(){return _xControlPoints;}
		std::shared_ptr<float1DReg> getYControlPoints(){return _yControlPoints;}

		// Forward / Adjoint
		void forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const;
		void adjoint(const bool add, std::shared_ptr<float3DReg> model, const std::shared_ptr<float3DReg> data) const;

};

#endif

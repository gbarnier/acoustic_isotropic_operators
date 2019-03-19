#include <string>
#include <iostream>
#include "interpBSpline2d.h"
#include <omp.h>
#include <vector>

// Contructor
interpBSpline2d::interpBSpline2d(int zOrder, int xOrder, std::shared_ptr<float1DReg> zControlPoints, std::shared_ptr<float1DReg> xControlPoints, axis zDataAxis, axis xDataAxis, int nzParamVector, int nxParamVector, int scaling, float zTolerance, float xTolerance, int fat){

    // B-spline parameters
    _zOrder = zOrder; // Order of interpolation in the z-direction
    _xOrder = xOrder; // Order of interpolation in the x-direction
    _scaling = scaling; // if = 1, compute and apply scaling to balance operator amplitudes

    // Model
    _zControlPoints = zControlPoints;
    _xControlPoints = xControlPoints;
    _nzModel = _zControlPoints->getHyper()->getAxis(1).n; // Number of control points in the z-direction
    _nxModel = _xControlPoints->getHyper()->getAxis(1).n; // Number of control points in the x-direction
    _nModel = _nzModel*_nxModel; // Total model size

    // Initialize the scale vector to 1.0
	_scaleVector=std::make_shared<float2DReg>(_nzModel, _nxModel);
    #pragma omp parallel for
	for (int ixModel=0; ixModel<_nxModel; ixModel++){
		for (int izModel=0; izModel<_nzModel; izModel++){
    		(*_scaleVector->_mat)[ixModel][izModel]=1.0;
		}
	}

    // Data
    _fat = fat;
    _zDataAxis = zDataAxis; // z-coordinates of data points assumed to be uniformly distributed
    _xDataAxis = xDataAxis; // x-coordinates of data points assumed to be uniformly distributed
    _nzData = _zDataAxis.n;
    _nxData = _xDataAxis.n;
    _nData =  _zDataAxis.n*_xDataAxis.n;
    _zData = std::make_shared<float1DReg>(_nzData);
    _xData = std::make_shared<float1DReg>(_nxData);

    // Compute z- and x-positions of data points on each axis
    for (int izData=0; izData<_nzData; izData++){
		(*_zData->_mat)[izData]=_zDataAxis.o+_zDataAxis.d*izData;
	}
	for (int ixData=0; ixData<_nxData; ixData++){
		(*_xData->_mat)[ixData]=_xDataAxis.o+_xDataAxis.d*ixData;
	}

    // Set the tolerance [km]
    _zTolerance=zTolerance*_zDataAxis.d;
    _xTolerance=xTolerance*_xDataAxis.d;

    // Number of points to evaluate in the parameter vectors
    _nzParamVector = nzParamVector;
    _nxParamVector = nxParamVector;

    // Build the knot vectors
    buildKnotVectors2d();

    // Compute parameter vectors
    _zParamVector = computeParamVectorZ();
    _xParamVector = computeParamVectorX();

    // Compute scale for amplitude balancing
    if(_scaling == 1) {
        _scaleVector = computeScaleVector();
    }

}

// Model mesh
std::shared_ptr<float1DReg> interpBSpline2d::getZMeshModel(){
    _zMeshModelVector = std::make_shared<float1DReg>(_nModel);

	#pragma omp parallel for collapse(2)
    for (int ix=0; ix<_nxModel; ix++){
        for (int iz=0; iz<_nzModel; iz++){
            int i=ix*_nzModel+iz;
            (*_zMeshModelVector->_mat)[i]=(*_zControlPoints->_mat)[iz];
        }
    }

    return _zMeshModelVector;
}
std::shared_ptr<float1DReg> interpBSpline2d::getXMeshModel(){
    _xMeshModelVector = std::make_shared<float1DReg>(_nModel);

    #pragma omp parallel for collapse(2)
    for (int ix=0; ix<_nxModel; ix++){
        for (int iz=0; iz<_nzModel; iz++){
            int i=ix*_nzModel+iz;
            (*_xMeshModelVector->_mat)[i]=(*_xControlPoints->_mat)[ix];
        }
    }

    return _xMeshModelVector;
}

// Data mesh
std::shared_ptr<float1DReg> interpBSpline2d::getZMeshData(){
    _zMeshDataVector = std::make_shared<float1DReg>(_nData);
    // Build mesh vectors (1D array) for the fine grid
    for (int ix=0; ix<_nxData; ix++){
        for (int iz=0; iz<_nzData; iz++){
            int i=ix*_nzData+iz;
            (*_zMeshDataVector->_mat)[i]=(*_zData->_mat)[iz];
        }
    }
    return _zMeshDataVector;
}
std::shared_ptr<float1DReg> interpBSpline2d::getXMeshData(){
    _xMeshDataVector = std::make_shared<float1DReg>(_nData);
    // Build mesh vectors (1D array) for the fine grid
    for (int ix=0; ix<_nxData; ix++){
        for (int iz=0; iz<_nzData; iz++){
            int i=ix*_nzData+iz;
            (*_xMeshDataVector->_mat)[i]=(*_xData->_mat)[ix];
        }
    }
    return _xMeshDataVector;
}

// Knot vectors for both directions
void interpBSpline2d::buildKnotVectors2d() {

    // Knots for the z-axis
    _nkz=_nzModel+_zOrder+1; // Number of knots
    _nkzSimple=_nkz-2*_zOrder; // Number of simple knots
    _okz = 0.0; // Position of FIRST knot
    _fkz = 1.0; // Position of LAST knot
    _dkz=(_fkz-_okz)/(_nkzSimple-1); // Knot sampling
    _dkz3=_dkz*_dkz*_dkz;
    _dkz2=_dkz*_dkz;
    _kzAxis = axis(_nkz, _okz, _dkz); // Knot axis
    _zKnots = std::make_shared<float1DReg>(_kzAxis);

    // Knots for the x-axis
    _nkx=_nxModel+_xOrder+1; // Number of knots
    _nkxSimple=_nkx-2*_xOrder; // Number of simple knots
    _okx = 0.0; // Position of FIRST knot
    _fkx = 1.0; // Position of LAST knot
    _dkx=(_fkx-_okx)/(_nkxSimple-1); // Knot sampling
    _dkx3=_dkx*_dkx*_dkx;
    _dkx2=_dkx*_dkx;
    _kxAxis = axis(_nkx, _okx, _dkx); // Knot axis
    _xKnots = std::make_shared<float1DReg>(_kxAxis);

    // Compute starting knots with multiplicity > 1 (if order>0)
    for (int ikz=0; ikz<_zOrder+1; ikz++){
		(*_zKnots->_mat)[ikz] = _okz;
	}
    // Compute knots with multiplicity 1
	for (int ikz=_zOrder+1; ikz<_nkz-_zOrder; ikz++){
        (*_zKnots->_mat)[ikz] = _okz+(ikz-_zOrder)*_dkz;
    }
    // Compute end knots with multiplicity > 1 (if order>0)
	for (int ikz=_nkz-_zOrder; ikz<_nkz; ikz++){
        (*_zKnots->_mat)[ikz] = _fkz;
    }

    // Compute starting knots with multiplicity > 1 (if order>0)
    for (int ikx=0; ikx<_xOrder+1; ikx++){
		(*_xKnots->_mat)[ikx] = _okx;
	}
    // Compute knots with multiplicity 1
	for (int ikx=_xOrder+1; ikx<_nkx-_xOrder; ikx++){
        (*_xKnots->_mat)[ikx] = _okx+(ikx-_xOrder)*_dkx;
    }
    // Compute end knots with multiplicity > 1 (if order>0)
	for (int ikx=_nkx-_xOrder; ikx<_nkx; ikx++){
        (*_xKnots->_mat)[ikx] = _fkx;
    }
}

// Compute parameter vector containing optimal parameters
std::shared_ptr<float1DReg> interpBSpline2d::computeParamVectorZ(){

    // Generate u (z-direction)
    int nu = _nzParamVector;
    float ou = _okz;
	float fu = _fkz;
	float du = (fu-ou)/(nu-1);
	nu=nu-1;

    std::shared_ptr<float1DReg> u(new float1DReg(nu));
    axis uAxis = axis(nu, ou, du);
    std::shared_ptr<float1DReg> paramVector(new float1DReg(_nzData));

    // Initialize param vector with -1
    #pragma omp parallel for
    for (int iData=0; iData<_nzData; iData++){
        (*paramVector->_mat)[iData]=-1.0;
    }

    // Loop over data space
	#pragma omp parallel for
    for (int izData=_fat; izData<_nzData-_fat; izData++){

        float error=100000;
        for (int iu=0; iu<nu; iu++){

            float uValue=ou+iu*du;
            float zInterp = 0;
            for (int izModel=0; izModel<_nzModel; izModel++){

                float sz=uValue-(*_zKnots->_mat)[izModel];
                float sz3=sz*sz*sz;
                float sz2=sz*sz;
                float zWeight=0.0;

                if ( sz>=0.0 && sz<4.0*_dkz ){
                    if (izModel==0){
                        if (sz>=0 && sz<_dkz){
                            zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
                        }
                    } else if (izModel==1){
                        if (sz>=0 && sz<_dkz){
                            zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
                        } else if (sz>=_dkz && sz<(2.0*_dkz)){
                            zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
                        }
                    } else if (izModel==2){
                        if (sz>=0 && sz<_dkz){
                            zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
                        } else if (sz>=_dkz && sz<(2.0*_dkz)){
                            zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
                        } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                            zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
                        }
                    } else if (izModel>=3 && izModel<_nzModel-3){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz3/(6.0*_dkz3);
                        } else if (sz>=_dkz && sz<(2.0*_dkz)){
                            zWeight = -sz3/(2.0*_dkz3) + 2.0*sz2/_dkz2 - 2.0*sz/_dkz + 2.0/3.0;
                        } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                            zWeight = 1/(2.0*_dkz3)*sz3 - 4.0/_dkz2*sz2 + 10.0*sz/_dkz -22.0/3.0;
                        } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
                            zWeight = -sz3/(6.0*_dkz3) + 2.0*sz2/_dkz2 - 8.0*sz/_dkz + 32.0/3.0;
                        }
                    } else if (izModel==_nzModel-3){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                        } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                            zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
                        } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
                            zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
                            zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
                        }
                    } else if (izModel==_nzModel-2){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
                        } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
                            zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
                            zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
                        }
                    } else if (izModel==_nzModel-1){
                        if (sz>=0.0 && sz<=_dkz){
                            zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
                        }
                    }
                }

                // Add contribution of model point
                zInterp+=zWeight*(*_zControlPoints->_mat)[izModel];

            }
            // Finished computing interpolated position for this u-value
            // Update the optimal u-value if interpolated point is clsoer to data point
            if (std::abs(zInterp-(*_zData->_mat)[izData]) < error) {
                error=std::abs(zInterp-(*_zData->_mat)[izData]);
                (*paramVector->_mat)[izData]=uValue;
            }
        }
        // Finished computing interpolated values for all u's
        if (std::abs(error)>_zTolerance){
            std::cout << "**** ERROR: Could not find a parameter for data point in the z-direction #" << izData << " " << (*_zData->_mat)[izData] << " [km]. Try increasing the number of samples! ****" << std::endl;
            std::cout << "Error = " << error << std::endl;
            std::cout << "Tolerance = " << _zTolerance << " [km]" << std::endl;
            assert(1==2);
        }
    }
    return paramVector;
}
std::shared_ptr<float1DReg> interpBSpline2d::computeParamVectorX(){

    // Generate u (z-direction)
    int nv = _nxParamVector;
    float ov = _okx;
	float fv = _fkx;
	float dv = (fv-ov)/(nv-1);
	nv=nv-1;

    std::shared_ptr<float1DReg> v(new float1DReg(nv));
    axis vAxis = axis(nv, ov, dv);
    std::shared_ptr<float1DReg> paramVector(new float1DReg(_nxData));

    // Initialize param vector with -1
    #pragma omp parallel for
    for (int iData=0; iData<_nxData; iData++){
        (*paramVector->_mat)[iData]=-1.0;
    }

    // Loop over data space
	#pragma omp parallel for
    for (int ixData=_fat; ixData<_nxData-_fat; ixData++){

        float error=100000;
        for (int iv=0; iv<nv; iv++){

            float vValue=ov+iv*dv;
            float xInterp = 0;
            for (int ixModel=0; ixModel<_nxModel; ixModel++){

                float sx=vValue-(*_xKnots->_mat)[ixModel];
                float sx3=sx*sx*sx;
                float sx2=sx*sx;
                float xWeight=0.0;

                if ( sx>=0.0 && sx<4.0*_dkx ){
                    if (ixModel==0){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
                        }
                    } else if (ixModel==1){
                        if (sx>=0 && sx<_dkx){
                            xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
                        }
                    } else if (ixModel==2){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
                        }
                    } else if (ixModel>=3 && ixModel<_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx3/(6.0*_dkx3);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight = -sx3/(2.0*_dkx3) + 2.0*sx2/(_dkx2) - 2.0*sx/_dkx + 2.0/3.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight = 1/(2.0*_dkx3)*sx3 - 4.0/_dkx2*sx2 + 10.0*sx/_dkx -22.0/3.0;
                        } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
                            xWeight = -sx3/(6.0*_dkx3) + 2.0*sx2/_dkx2 - 8.0*sx/_dkx + 32.0/3.0;
                        }
                    } else if (ixModel==_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
                        } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
                            xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
                            xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
                        }
                    } else if (ixModel==_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
                            xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
                            xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
                        }
                    } else if (ixModel==_nxModel-1){
                        if (sx>=0.0 && sx<=_dkx){
                            xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
                        }
                    }

                    // Add contribution of model point
                    xInterp+=xWeight*(*_xControlPoints->_mat)[ixModel];
                }
            }
            // Finished computing interpolated position for this u-value
            // Update the optimal u-value if interpolated point is clsoer to data point
            if (std::abs(xInterp-(*_xData->_mat)[ixData]) < error) {
                error=std::abs(xInterp-(*_xData->_mat)[ixData]);
                (*paramVector->_mat)[ixData]=vValue;
            }
        }
        // Finished computing interpolated values for all u's
        if (std::abs(error)>_xTolerance){
            std::cout << "**** ERROR: Could not find a parameter for data point in the x-direction #" << ixData << " " << (*_xData->_mat)[ixData]<< " [km]. Try increasing the number of samples! ****" << std::endl;
            std::cout << "Error = " << error << std::endl;
            std::cout << "Tolerance = " << _xTolerance << " [km]" << std::endl;
            assert(1==2);
        }
    }
    return paramVector;
}

// Scaling vector
std::shared_ptr<float2DReg> interpBSpline2d::computeScaleVector(){

    // Variables declaration
    float uValue, vValue, zWeight, xWeight;
    std::shared_ptr<float2DReg> scaleVector, scaleVectorData;
    scaleVector = std::make_shared<float2DReg>(_nzModel, _nxModel);
    scaleVectorData = std::make_shared<float2DReg>(_nzData, _nxData);
    scaleVectorData->scale(0.0);
    scaleVector->set(1.0);

    // Apply one forward
    forward(false, scaleVector, scaleVectorData);

    // Apply one adjoint
    adjoint(false, scaleVector, scaleVectorData);

    // Compute scaling
    #pragma omp parallel for collapse(2)
    for (int ixModel=0; ixModel<_nxModel; ixModel++){
        for (int izModel=0; izModel<_nzModel; izModel++){
            (*scaleVector->_mat)[ixModel][izModel]=1.0/sqrt((*scaleVector->_mat)[ixModel][izModel]);
        }
    }

    return scaleVector;
}

// Forward
void interpBSpline2d::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const {

    // Forward: Coarse grid to fine grid
    // Model can be on an irregular grid
	if (!add) data->scale(0.0);

    // Loop over data (fine sampling grid)
	#pragma omp parallel for collapse(2)
    for (int ixData=_fat; ixData<_nxData-_fat; ixData++){
        for (int izData=_fat; izData<_nzData-_fat; izData++){

            float uValue = (*_zParamVector->_mat)[izData];
            float vValue = (*_xParamVector->_mat)[ixData];

            for (int ixModel=0; ixModel<_nxModel; ixModel++){

                float sx=vValue-(*_xKnots->_mat)[ixModel];
                float sx3=sx*sx*sx;
                float sx2=sx*sx;
                float xWeight=0.0;

                if( sx>=0.0 && sx<4.0*_dkx ){

                    if (ixModel==0){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
                        }
                    } else if (ixModel==1){
                        if (sx>=0 && sx<_dkx){
                            xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
                        }
                    } else if (ixModel==2){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
                        }
                    } else if (ixModel>=3 && ixModel<_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx3/(6.0*_dkx3);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight = -sx3/(2.0*_dkx3) + 2.0*sx2/(_dkx2) - 2.0*sx/_dkx + 2.0/3.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight = 1/(2.0*_dkx3)*sx3 - 4.0/_dkx2*sx2 + 10.0*sx/_dkx -22.0/3.0;
                        } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
                            xWeight = -sx3/(6.0*_dkx3) + 2.0*sx2/_dkx2 - 8.0*sx/_dkx + 32.0/3.0;
                        }
                    } else if (ixModel==_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
                        } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
                            xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
                            xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
                        }
                    } else if (ixModel==_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
                            xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
                            xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
                        }
                    } else if (ixModel==_nxModel-1){
                        if (sx>=0.0 && sx<=_dkx){
                            xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
                        }
                    }

                    for (int izModel=0; izModel<_nzModel; izModel++){

                        float sz=uValue-(*_zKnots->_mat)[izModel];
                        float sz3=sz*sz*sz;
                        float sz2=sz*sz;
                        float zWeight=0.0;

                        if( sz>=0.0 && sz<4.0*_dkz ){

                            if (izModel==0){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
                                }
                            } else if (izModel==1){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
                                }
                            } else if (izModel==2){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
                                }
                            } else if (izModel>=3 && izModel<_nzModel-3){
                                if (sz>=0.0 && sz<_dkz){
                                    // zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                                    zWeight=sz3/(6.0*_dkz3);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    // zWeight = -sz*sz*sz/(2.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 2.0*sz/_dkz + 2.0/3.0;
                                    zWeight = -sz3/(2.0*_dkz3) + 2.0*sz2/_dkz2 - 2.0*sz/_dkz + 2.0/3.0;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    // zWeight = 1/(2.0*_dkz*_dkz*_dkz)*sz*sz*sz - 4.0/(_dkz*_dkz)*sz*sz + 10.0*sz/_dkz -22.0/3.0;
                                    zWeight = 1/(2.0*_dkz3)*sz3 - 4.0/_dkz2*sz2 + 10.0*sz/_dkz -22.0/3.0;
                                } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
                                    // zWeight = -sz*sz*sz/(6.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 8.0*sz/_dkz + 32.0/3.0;
                                    zWeight = -sz3/(6.0*_dkz3) + 2.0*sz2/_dkz2 - 8.0*sz/_dkz + 32.0/3.0;
                                }

                            } else if (izModel==_nzModel-3){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                                    zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
                                } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
                                    zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
                                    zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
                                }
                            } else if (izModel==_nzModel-2){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
                                    zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
                                    zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
                                }
                            } else if (izModel==_nzModel-1){
                                if (sz>=0.0 && sz<=_dkz){
                                    zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
                                }
                            }

                            // Add contribution to interpolated value (data)
                            (*data->_mat)[ixData][izData] += xWeight*zWeight*(*_scaleVector->_mat)[ixModel][izModel]*(*model->_mat)[ixModel][izModel];
                        }
                    }
                }
            }
        }
    }
}

// Adjoint
void interpBSpline2d::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const {

    // Adjoint: Fine grid to coarse grid
    // Model can be on an irregular grid
    if (!add) model->scale(0.0);

    #pragma omp parallel for collapse(2)
    for (int ixModel=0; ixModel<_nxModel; ixModel++){
        for (int izModel=0; izModel<_nzModel; izModel++){

            for (int ixData=_fat; ixData<_nxData-_fat; ixData++){
                float vValue = (*_xParamVector->_mat)[ixData];
                float sx=vValue-(*_xKnots->_mat)[ixModel];
                float sx3=sx*sx*sx;
                float sx2=sx*sx;
                float xWeight=0.0;

                if( sx>=0.0 && sx<4.0*_dkx ){

                    if (ixModel==0){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
                        }
                    } else if (ixModel==1){
                        if (sx>=0 && sx<_dkx){
                            xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
                        }
                    } else if (ixModel==2){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
                        }
                    } else if (ixModel>=3 && ixModel<_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx3/(6.0*_dkx3);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight = -sx3/(2.0*_dkx3) + 2.0*sx2/(_dkx2) - 2.0*sx/_dkx + 2.0/3.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight = 1/(2.0*_dkx3)*sx3 - 4.0/_dkx2*sx2 + 10.0*sx/_dkx -22.0/3.0;
                        } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
                            xWeight = -sx3/(6.0*_dkx3) + 2.0*sx2/_dkx2 - 8.0*sx/_dkx + 32.0/3.0;
                        }
                    } else if (ixModel==_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
                        } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
                            xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
                            xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
                        }
                    } else if (ixModel==_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
                            xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
                            xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
                        }
                    } else if (ixModel==_nxModel-1){
                        if (sx>=0.0 && sx<=_dkx){
                            xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
                        }
                    }

                    for (int izData=_fat; izData<_nzData-_fat; izData++){
                        float uValue = (*_zParamVector->_mat)[izData];
                        float sz=uValue-(*_zKnots->_mat)[izModel];
                        float sz3=sz*sz*sz;
                        float sz2=sz*sz;
                        float zWeight=0.0;

                        if( sz>=0.0 && sz<4.0*_dkz ){

                            if (izModel==0){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
                                }
                            } else if (izModel==1){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
                                }
                            } else if (izModel==2){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
                                }
                            } else if (izModel>=3 && izModel<_nzModel-3){
                                if (sz>=0.0 && sz<_dkz){
                                        zWeight=sz3/(6.0*_dkz3);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight = -sz3/(2.0*_dkz3) + 2.0*sz2/_dkz2 - 2.0*sz/_dkz + 2.0/3.0;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    zWeight = 1/(2.0*_dkz3)*sz3 - 4.0/_dkz2*sz2 + 10.0*sz/_dkz -22.0/3.0;
                                } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
                                    zWeight = -sz3/(6.0*_dkz3) + 2.0*sz2/_dkz2 - 8.0*sz/_dkz + 32.0/3.0;
                                }
                            } else if (izModel==_nzModel-3){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                                    zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
                                } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
                                    zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
                                    zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
                                }
                            } else if (izModel==_nzModel-2){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
                                    zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
                                    zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
                                }
                            } else if (izModel==_nzModel-1){
                                if (sz>=0.0 && sz<=_dkz){
                                    zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
                                }
                            }

                            (*model->_mat)[ixModel][izModel] += xWeight*zWeight*(*_scaleVector->_mat)[ixModel][izModel]*(*data->_mat)[ixData][izData];
                        }
                    }
                }
            }
        }
    }
}

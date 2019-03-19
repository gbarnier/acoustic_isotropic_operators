#include <string>
#include <float3DReg.h>
#include <iostream>
#include "interpBSpline3d.h"
#include <omp.h>
#include <vector>

// Contructor
interpBSpline3d::interpBSpline3d(int zOrder, int xOrder, int yOrder, std::shared_ptr<float1DReg> zControlPoints, std::shared_ptr<float1DReg> xControlPoints, std::shared_ptr<float1DReg> yControlPoints, axis zDataAxis, axis xDataAxis, axis yDataAxis, int nzParamVector, int nxParamVector, int nyParamVector, int scaling, float zTolerance, float xTolerance, float yTolerance, int zFat, int xFat, int yFat){

    // B-spline parameters
    _zOrder = zOrder; // Order of interpolation in the z-direction
    _xOrder = xOrder; // Order of interpolation in the x-direction
    _yOrder = yOrder; // Order of interpolation in the x-direction
    _scaling = scaling; // if = 1, compute and apply scaling to balance operator amplitudes

    // Model
    _zControlPoints = zControlPoints;
    _xControlPoints = xControlPoints;
    _yControlPoints = yControlPoints;
    _nzModel = _zControlPoints->getHyper()->getAxis(1).n; // Number of control points in the z-direction
    _nxModel = _xControlPoints->getHyper()->getAxis(1).n; // Number of control points in the x-direction
    _nyModel = _yControlPoints->getHyper()->getAxis(1).n; // Number of control points in the y-direction
    _nModel = _nzModel*_nxModel*_nyModel; // Total model size

    // Initialize the scale vector to 1.0
	_scaleVector=std::make_shared<float3DReg>(_nzModel, _nxModel, _nyModel);
    #pragma omp parallel for
	for (int iyModel=0; iyModel<_nyModel; iyModel++){
		for (int ixModel=0; ixModel<_nxModel; ixModel++){
			for (int izModel=0; izModel<_nzModel; izModel++){
        		(*_scaleVector->_mat)[iyModel][ixModel][izModel]=1.0;
    		}
		}
	}

    // Data
    _zFat = zFat;
    _xFat = xFat;
    _yFat = yFat;
    _zDataAxis = zDataAxis; // z-coordinates of data points assumed to be uniformly distributed
    _xDataAxis = xDataAxis; // x-coordinates of data points assumed to be uniformly distributed
    _yDataAxis = yDataAxis; // y-coordinates of data points assumed to be uniformly distributed
    _nzData = _zDataAxis.n;
    _nxData = _xDataAxis.n;
    _nyData = _yDataAxis.n;
    _nData =  _nzData*_nxData*_nyData;

	_zData=std::make_shared<float1DReg>(_nzData);
	_xData=std::make_shared<float1DReg>(_nxData);
	_yData=std::make_shared<float1DReg>(_nyData);
	for (int izData=0; izData<_nzData; izData++){
		(*_zData->_mat)[izData]=_zDataAxis.o+_zDataAxis.d*izData;
	}
	for (int ixData=0; ixData<_nxData; ixData++){
		(*_xData->_mat)[ixData]=_xDataAxis.o+_xDataAxis.d*ixData;
	}
	for (int iyData=0; iyData<_nyData; iyData++){
		(*_yData->_mat)[iyData]=_yDataAxis.o+_yDataAxis.d*iyData;
	}

    // Set the tolerance [km]
    _zTolerance=zTolerance*_zDataAxis.d;
    _xTolerance=xTolerance*_xDataAxis.d;
    _yTolerance=yTolerance*_yDataAxis.d;

    // Number of points to evaluate in the parameter vectors
    _nzParamVector = nzParamVector;
    _nxParamVector = nxParamVector;
    _nyParamVector = nyParamVector;


    // Build the knot vectors
    buildKnotVectors3d();

    // Compute parameter vectors
    _zParamVector = computeParamVectorZ();
    _xParamVector = computeParamVectorX();
    _yParamVector = computeParamVectorY();

    computeZDataIndex();
    computeXDataIndex();
    computeYDataIndex();
    computeZModelIndex();
    computeXModelIndex();
    computeYModelIndex();

    // Compute scale for amplitude balancing
    if (_scaling == 1) {_scaleVector = computeScaleVector();}

}

// Model mesh
std::shared_ptr<float1DReg> interpBSpline3d::getZMeshModel(){

    _zMeshModelVector = std::make_shared<float1DReg>(_nModel);

	#pragma omp parallel for collapse(3)
	for (int iy=0; iy<_nyModel; iy++){
	    for (int ix=0; ix<_nxModel; ix++){
	        for (int iz=0; iz<_nzModel; iz++){
	            int i=iy*_nzModel*_nxModel+ix*_nzModel+iz;
	            (*_zMeshModelVector->_mat)[i]=(*_zControlPoints->_mat)[iz];
	        }
	    }
	}
    return _zMeshModelVector;
}
std::shared_ptr<float1DReg> interpBSpline3d::getXMeshModel(){
    _xMeshModelVector = std::make_shared<float1DReg>(_nModel);

    #pragma omp parallel for collapse(3)
	for (int iy=0; iy<_nyModel; iy++){
	    for (int ix=0; ix<_nxModel; ix++){
	        for (int iz=0; iz<_nzModel; iz++){
	            int i=iy*_nzModel*_nxModel+ix*_nzModel+iz;
	            (*_xMeshModelVector->_mat)[i]=(*_xControlPoints->_mat)[ix];
	        }
	    }
	}
    return _xMeshModelVector;
}
std::shared_ptr<float1DReg> interpBSpline3d::getYMeshModel(){
    _yMeshModelVector = std::make_shared<float1DReg>(_nModel);

    #pragma omp parallel for collapse(3)
	for (int iy=0; iy<_nyModel; iy++){
	    for (int ix=0; ix<_nxModel; ix++){
	        for (int iz=0; iz<_nzModel; iz++){
	            int i=iy*_nzModel*_nxModel+ix*_nzModel+iz;
	            (*_yMeshModelVector->_mat)[i]=(*_yControlPoints->_mat)[iy];
	        }
	    }
	}
    return _yMeshModelVector;
}

// Data mesh
std::shared_ptr<float1DReg> interpBSpline3d::getZMeshData(){
    _zMeshDataVector = std::make_shared<float1DReg>(_nData);
    // Build mesh vectors (1D array) for the fine grid
	for (int iy=0; iy<_nyData; iy++){
	    for (int ix=0; ix<_nxData; ix++){
	        for (int iz=0; iz<_nzData; iz++){
	            int i=iy*_nzData*_nxData+ix*_nzData+iz;
	            (*_zMeshDataVector->_mat)[i]=_zDataAxis.o+iz*_zDataAxis.d;
	        }
	    }
	}
    return _zMeshDataVector;
}
std::shared_ptr<float1DReg> interpBSpline3d::getXMeshData(){
    _xMeshDataVector = std::make_shared<float1DReg>(_nData);
    // Build mesh vectors (1D array) for the fine grid
	for (int iy=0; iy<_nyData; iy++){
	    for (int ix=0; ix<_nxData; ix++){
			float x=_xDataAxis.o+ix*_xDataAxis.d;
	        for (int iz=0; iz<_nzData; iz++){
	            int i=iy*_nzData*_nxData+ix*_nzData+iz;
				(*_xMeshDataVector->_mat)[i]=x;
	        }
	    }
	}
    return _xMeshDataVector;
}
std::shared_ptr<float1DReg> interpBSpline3d::getYMeshData(){
    _yMeshDataVector = std::make_shared<float1DReg>(_nData);
    // Build mesh vectors (1D array) for the fine grid
	for (int iy=0; iy<_nyData; iy++){
		float y=_yDataAxis.o+iy*_yDataAxis.d;
	    for (int ix=0; ix<_nxData; ix++){
	        for (int iz=0; iz<_nzData; iz++){
				int i=iy*_nzData*_nxData+ix*_nzData+iz;
				(*_yMeshDataVector->_mat)[i]=y;
	        }
	    }
	}
    return _yMeshDataVector;
}

// Compute model index
void interpBSpline3d::computeZDataIndex(){
    _zDataIndex.resize(_nzData);
    #pragma omp parallel for
    for (int izData=_zFat; izData<_nzData-_zFat; izData++){
        double uValue = (*_zParamVector->_mat)[izData];
        for (int izModel=0; izModel<_nzModel; izModel++){
            double sz=uValue-(*_zKnots->_mat)[izModel];
            if ( sz>=0.0 && sz<4.0*_dkz ){
                _zDataIndex[izData].push_back(izModel);
            }
        }
    }
}
void interpBSpline3d::computeXDataIndex(){
    _xDataIndex.resize(_nxData);
    #pragma omp parallel for
    for (int ixData=_xFat; ixData<_nxData-_xFat; ixData++){
        double vValue = (*_xParamVector->_mat)[ixData];
        for (int ixModel=0; ixModel<_nxModel; ixModel++){
            double sx=vValue-(*_xKnots->_mat)[ixModel];
            if ( sx>=0.0 && sx<4.0*_dkx ){
                _xDataIndex[ixData].push_back(ixModel);
            }
        }
    }
}
void interpBSpline3d::computeYDataIndex(){
    _yDataIndex.resize(_nyData);
    #pragma omp parallel for
    for (int iyData=_yFat; iyData<_nyData-_yFat; iyData++){
        double wValue = (*_yParamVector->_mat)[iyData];
        for (int iyModel=0; iyModel<_nyModel; iyModel++){
            double sy=wValue-(*_yKnots->_mat)[iyModel];
            if ( sy>=0.0 && sy<4.0*_dky ){
                _yDataIndex[iyData].push_back(iyModel);
            }
        }
    }
}

void interpBSpline3d::computeZModelIndex(){
    _zModelIndex.resize(_nzModel);
    #pragma omp parallel for
    for (int izModel=0; izModel<_nzModel; izModel++){
        for (int izData=_zFat; izData<_nzData-_zFat; izData++){
            double uValue = (*_zParamVector->_mat)[izData];
            double sz=uValue-(*_zKnots->_mat)[izModel];
            if ( sz>=0.0 && sz<4.0*_dkz ){
                _zModelIndex[izModel].push_back(izData);
            }
        }
    }
}
void interpBSpline3d::computeXModelIndex(){
    _xModelIndex.resize(_nxModel);
    #pragma omp parallel for
    for (int ixModel=0; ixModel<_nxModel; ixModel++){
        for (int ixData=_xFat; ixData<_nxData-_xFat; ixData++){
            double vValue = (*_xParamVector->_mat)[ixData];
            double sx=vValue-(*_xKnots->_mat)[ixModel];
            if ( sx>=0.0 && sx<4.0*_dkx ){
                _xModelIndex[ixModel].push_back(ixData);
            }
        }
    }
}
void interpBSpline3d::computeYModelIndex(){
    _yModelIndex.resize(_nyModel);
    #pragma omp parallel for
    for (int iyModel=0; iyModel<_nyModel; iyModel++){
        for (int iyData=_yFat; iyData<_nyData-_yFat; iyData++){
            double wValue = (*_yParamVector->_mat)[iyData];
            double sy=wValue-(*_yKnots->_mat)[iyModel];
            if ( sy>=0.0 && sy<4.0*_dky ){
                _yModelIndex[iyModel].push_back(iyData);
            }
        }
    }
}

// // Knot vectors for both directions
void interpBSpline3d::buildKnotVectors3d(){

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

    // Knots for the y-axis
    _nky=_nyModel+_yOrder+1; // Number of knots
    _nkySimple=_nky-2*_yOrder; // Number of simple knots
    _oky = 0.0; // Position of FIRST knot
    _fky = 1.0; // Position of LAST knot
    _dky=(_fky-_oky)/(_nkySimple-1); // Knot sampling
    _dky3=_dky*_dky*_dky;
    _dky2=_dky*_dky;
    _kyAxis = axis(_nky, _oky, _dky); // Knot axis
    _yKnots = std::make_shared<float1DReg>(_kyAxis);

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

    // Compute starting knots with multiplicity > 1 (if order>0)
    for (int iky=0; iky<_yOrder+1; iky++){
		(*_yKnots->_mat)[iky] = _oky;
	}
    // Compute knots with multiplicity 1
	for (int iky=_yOrder+1; iky<_nky-_yOrder; iky++){
        (*_yKnots->_mat)[iky] = _oky+(iky-_yOrder)*_dky;
    }
    // Compute end knots with multiplicity > 1 (if order>0)
	for (int iky=_nky-_yOrder; iky<_nky; iky++){
        (*_yKnots->_mat)[iky] = _fky;
    }
}

// Compute parameter vector containing optimal parameters
std::shared_ptr<float1DReg> interpBSpline3d::computeParamVectorZ(){

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
    // #pragma omp parallel for
    for (int iData=0; iData<_nzData; iData++){
        (*paramVector->_mat)[iData]=-1.0;
    }

    // Loop over data space
	// #pragma omp parallel for
    for (int izData=_zFat; izData<_nzData-_zFat; izData++){
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
std::shared_ptr<float1DReg> interpBSpline3d::computeParamVectorX(){

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
    for (int ixData=_xFat; ixData<_nxData-_xFat; ixData++){

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
std::shared_ptr<float1DReg> interpBSpline3d::computeParamVectorY(){

    // Generate u (z-direction)
    int nw = _nyParamVector;
    float ow = _oky;
	float fw = _fky;
	float dw = (fw-ow)/(nw-1);
	nw=nw-1;

    std::shared_ptr<float1DReg> w(new float1DReg(nw));
    axis wAxis = axis(nw, ow, dw);
    std::shared_ptr<float1DReg> paramVector(new float1DReg(_nyData));

    // Initialize param vector with -1
    // #pragma omp parallel for
    for (int iData=0; iData<_nyData; iData++){
        (*paramVector->_mat)[iData]=-1.0;
    }

    // Loop over data space
	#pragma omp parallel for
    for (int iyData=_yFat; iyData<_nyData-_yFat; iyData++){
        float error=100000;
        for (int iw=0; iw<nw; iw++){

            float wValue=ow+iw*dw;
            float yInterp = 0;
            for (int iyModel=0; iyModel<_nyModel; iyModel++){

                float sy=wValue-(*_yKnots->_mat)[iyModel];
                float sy3=sy*sy*sy;
                float sy2=sy*sy;
                float yWeight=0.0;

                if ( sy>=0.0 && sy<4.0*_dky ){
                    if (iyModel==0){
                        if (sy>=0 && sy<_dky){
                            yWeight=-sy*sy*sy/(_dky*_dky*_dky)+3.0*sy*sy/(_dky*_dky)-3.0*sy/_dky+1.0;
                        }
                    } else if (iyModel==1){
                        if (sy>=0 && sy<_dky){
                            yWeight=7.0*sy*sy*sy/(4.0*_dky*_dky*_dky)-9.0*sy*sy/(2.0*_dky*_dky)+3.0*sy/_dky;
                        } else if (sy>=_dky && sy<(2.0*_dky)){
                            yWeight=-sy*sy*sy/(4.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-3.0*sy/_dky+2.0;
                        }
                    } else if (iyModel==2){
                        if (sy>=0 && sy<_dky){
                            yWeight=-11.0*sy*sy*sy/(12.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky);
                        } else if (sy>=_dky && sy<(2.0*_dky)){
                            yWeight=7.0*sy*sy*sy/(12.0*_dky*_dky*_dky)-3*sy*sy/(_dky*_dky)+9.0*sy/(2.0*_dky)-3.0/2.0;
                        } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
                            yWeight=-sy*sy*sy/(6.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-9.0*sy/(2.0*_dky)+9.0/2.0;
                        }
                    } else if (iyModel>=3 && iyModel<_nyModel-3){
                        if (sy>=0.0 && sy<_dky){
                            yWeight=sy3/(6.0*_dky3);
                        } else if (sy>=_dky && sy<(2.0*_dky)){
                            yWeight = -sy3/(2.0*_dky3) + 2.0*sy2/_dky2 - 2.0*sy/_dky + 2.0/3.0;
                        } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
                            yWeight = 1/(2.0*_dky3)*sy3 - 4.0/_dky2*sy2 + 10.0*sy/_dky -22.0/3.0;
                        } else if (sy>=(3.0*_dky) && sy<(4.0*_dky)){
                            yWeight = -sy3/(6.0*_dky3) + 2.0*sy2/_dky2 - 8.0*sy/_dky + 32.0/3.0;
                        }
                    } else if (iyModel==_nyModel-3){
                        if (sy>=0.0 && sy<_dky){
                            yWeight=sy*sy*sy/(6.0*_dky*_dky*_dky);
                        } else if(sy>=_dky && sy<(2.0*_dky)) {
                            yWeight=-sy*sy*sy/(3.0*_dky*_dky*_dky)+sy*sy/(_dky*_dky)-sy/(2*_dky)+(3.0/2.0-sy/(2.0*_dky))*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky);
                        } else if(sy>=(2.0*_dky) && sy<=(3.0*_dky)) {
                            yWeight=sy/(3.0*_dky)*(sy*sy/(2.0*_dky*_dky)-3*sy/_dky+9.0/2.0);
                            yWeight+=(3.0/2.0-sy/(2.0*_dky))*(-3*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky)+4*(sy-_dky)/_dky-2.0);
                        }
                    } else if (iyModel==_nyModel-2){
                        if (sy>=0.0 && sy<_dky){
                            yWeight=sy*sy*sy/(4.0*_dky*_dky*_dky);
                        } else if(sy>=_dky && sy<=(2.0*_dky)) {
                            yWeight=sy/(2.0*_dky)*(-3.0*sy*sy/(2.0*_dky*_dky)+4.0*sy/_dky-2.0);
                            yWeight+=(2.0-sy/_dky)*(sy-_dky)*(sy-_dky)/(_dky*_dky);
                        }
                    } else if (iyModel==_nyModel-1){
                        if (sy>=0.0 && sy<=_dky){
                            yWeight=sy*sy*sy/(_dky*_dky*_dky);
                        }
                    }
                }

                // Add contribution of model point
                yInterp+=yWeight*(*_yControlPoints->_mat)[iyModel];

            }
            // Finished computing interpolated position for this u-value
            // Update the optimal u-value if interpolated point is clsoer to data point

            if (std::abs(yInterp-(*_yData->_mat)[iyData]) < error) {
                error=std::abs(yInterp-(*_yData->_mat)[iyData]);
                (*paramVector->_mat)[iyData]=wValue;
            }

        }

        // Finished computing interpolated values for all u's
        if (std::abs(error)>_yTolerance){
            std::cout << "**** ERROR: Could not find a parameter for data point in the y-direction #" << iyData << " " << (*_yData->_mat)[iyData] << " [km]. Try increasing the number of samples! ****" << std::endl;
            std::cout << "Error = " << error << std::endl;
            std::cout << "Value = " << (*_yData->_mat)[iyData] << std::endl;
            std::cout << "Tolerance = " << _yTolerance << " [km]" << std::endl;
            assert(1==2);
        }
    }
    return paramVector;
}

// Scaling vector
std::shared_ptr<float3DReg> interpBSpline3d::computeScaleVector(){

	// Variables declaration
    float uValue, vValue, wValue, zWeight, xWeight, yWeight;
    std::shared_ptr<float3DReg> scaleVector, scaleVectorData;
    scaleVector = std::make_shared<float3DReg>(_nzModel, _nxModel, _nyModel);
    scaleVectorData = std::make_shared<float3DReg>(_nzData, _nxData, _nyData);
    scaleVector->scale(0.0);
    scaleVectorData->scale(0.0);

	// Apply one forward
    forward(false, _scaleVector, scaleVectorData);

	// Apply one adjoint
    adjoint(false, scaleVector, scaleVectorData);

    // Compute scaling
    #pragma omp parallel for collapse(3)
	for (int iyModel=0; iyModel<_nyModel; iyModel++){
		for (int ixModel=0; ixModel<_nxModel; ixModel++){
        	for (int izModel=0; izModel<_nzModel; izModel++){
            	(*scaleVector->_mat)[iyModel][ixModel][izModel]=1.0/sqrt((*scaleVector->_mat)[iyModel][ixModel][izModel]);
        	}
    	}
	}

    return scaleVector;

}

// Forward
// void interpBSpline3d::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const {
//
//     // Forward: Coarse grid to fine grid
//     // Model can be on an irregular grid
// 	if (!add) data->scale(0.0);
//
//     // Loop over data (fine sampling grid)
// 	#pragma omp parallel for collapse(3)
// 	for (int iyData=_yFat; iyData<_nyData-_yFat; iyData++){
//     	for (int ixData=_xFat; ixData<_nxData-_xFat; ixData++){
//         	for (int izData=_zFat; izData<_nzData-_zFat; izData++){
//
//             	float uValue = (*_zParamVector->_mat)[izData];
//             	float vValue = (*_xParamVector->_mat)[ixData];
// 				float wValue = (*_yParamVector->_mat)[iyData];
//
//             	for (int iyModel=0; iyModel<_nyModel; iyModel++){
// 					float sy=wValue-(*_yKnots->_mat)[iyModel];
// 					float sy3=sy*sy*sy;
// 					float sy2=sy*sy;
// 					float yWeight=0.0;
//
//                 	if( sy>=0.0 && sy<4.0*_dky ){
//
// 						///////////// Compute weight in y-direction ////////////
//
// 						if (iyModel==0){
// 	                        if (sy>=0 && sy<_dky){
// 	                            yWeight=-sy*sy*sy/(_dky*_dky*_dky)+3.0*sy*sy/(_dky*_dky)-3.0*sy/_dky+1.0;
// 	                        }
// 	                    } else if (iyModel==1){
// 	                        if (sy>=0 && sy<_dky){
// 	                            yWeight=7.0*sy*sy*sy/(4.0*_dky*_dky*_dky)-9.0*sy*sy/(2.0*_dky*_dky)+3.0*sy/_dky;
// 	                        } else if (sy>=_dky && sy<(2.0*_dky)){
// 	                            yWeight=-sy*sy*sy/(4.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-3.0*sy/_dky+2.0;
// 	                        }
// 	                    } else if (iyModel==2){
// 	                        if (sy>=0 && sy<_dky){
// 	                            yWeight=-11.0*sy*sy*sy/(12.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky);
// 	                        } else if (sy>=_dky && sy<(2.0*_dky)){
// 	                            yWeight=7.0*sy*sy*sy/(12.0*_dky*_dky*_dky)-3*sy*sy/(_dky*_dky)+9.0*sy/(2.0*_dky)-3.0/2.0;
// 	                        } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
// 	                            yWeight=-sy*sy*sy/(6.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-9.0*sy/(2.0*_dky)+9.0/2.0;
// 	                        }
// 	                    } else if (iyModel>=3 && iyModel<_nyModel-3){
// 	                        if (sy>=0.0 && sy<_dky){
// 	                            yWeight=sy3/(6.0*_dky3);
// 	                        } else if (sy>=_dky && sy<(2.0*_dky)){
// 	                            yWeight = -sy3/(2.0*_dky3) + 2.0*sy2/(_dky2) - 2.0*sy/_dky + 2.0/3.0;
// 	                        } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
// 	                            yWeight = 1/(2.0*_dky3)*sy3 - 4.0/_dky2*sy2 + 10.0*sy/_dky -22.0/3.0;
// 	                        } else if (sy>=(3.0*_dky) && sy<(4.0*_dky)){
// 	                            yWeight = -sy3/(6.0*_dky3) + 2.0*sy2/_dky2 - 8.0*sy/_dky + 32.0/3.0;
// 	                        }
// 	                    } else if (iyModel==_nyModel-3){
// 	                        if (sy>=0.0 && sy<_dky){
// 	                            yWeight=sy*sy*sy/(6.0*_dky*_dky*_dky);
// 	                        } else if(sy>=_dky && sy<(2.0*_dky)) {
// 	                            yWeight=-sy*sy*sy/(3.0*_dky*_dky*_dky)+sy*sy/(_dky*_dky)-sy/(2*_dky)+(3.0/2.0-sy/(2.0*_dky))*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky);
// 	                        } else if(sy>=(2.0*_dky) && sy<=(3.0*_dky)) {
// 	                            yWeight=sy/(3.0*_dky)*(sy*sy/(2.0*_dky*_dky)-3*sy/_dky+9.0/2.0);
// 	                            yWeight+=(3.0/2.0-sy/(2.0*_dky))*(-3*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky)+4*(sy-_dky)/_dky-2.0);
// 	                        }
// 	                    } else if (iyModel==_nyModel-2){
// 	                        if (sy>=0.0 && sy<_dky){
// 	                            yWeight=sy*sy*sy/(4.0*_dky*_dky*_dky);
// 	                        } else if(sy>=_dky && sy<=(2.0*_dky)) {
// 	                            yWeight=sy/(2.0*_dky)*(-3.0*sy*sy/(2.0*_dky*_dky)+4.0*sy/_dky-2.0);
// 	                            yWeight+=(2.0-sy/_dky)*(sy-_dky)*(sy-_dky)/(_dky*_dky);
// 	                        }
// 	                    } else if (iyModel==_nyModel-1){
// 	                        if (sy>=0.0 && sy<=_dky){
// 	                            yWeight=sy*sy*sy/(_dky*_dky*_dky);
// 	                        }
// 	                    }
// 						////////////////////////////////////////////////////////
//
// 						///////////// Compute weight in x-direction ////////////
// 						for (int ixModel=0; ixModel<_nxModel; ixModel++){
//
// 		                	float sx=vValue-(*_xKnots->_mat)[ixModel];
// 		                	float sx3=sx*sx*sx;
// 		                	float sx2=sx*sx;
// 		                	float xWeight=0.0;
//
//                 			if( sx>=0.0 && sx<4.0*_dkx ){
//
// 			                    if (ixModel==0){
// 			                        if (sx>=0 && sx<_dkx){
// 			                            xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
// 			                        }
// 			                    } else if (ixModel==1){
// 			                        if (sx>=0 && sx<_dkx){
// 			                            xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
// 			                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
// 			                            xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
// 			                        }
// 			                    } else if (ixModel==2){
// 			                        if (sx>=0 && sx<_dkx){
// 			                            xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
// 			                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
// 			                            xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
// 			                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
// 			                            xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
// 			                        }
// 			                    } else if (ixModel>=3 && ixModel<_nxModel-3){
// 			                        if (sx>=0.0 && sx<_dkx){
// 			                            xWeight=sx3/(6.0*_dkx3);
// 			                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
// 			                            xWeight = -sx3/(2.0*_dkx3) + 2.0*sx2/(_dkx2) - 2.0*sx/_dkx + 2.0/3.0;
// 			                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
// 			                            xWeight = 1/(2.0*_dkx3)*sx3 - 4.0/_dkx2*sx2 + 10.0*sx/_dkx -22.0/3.0;
// 			                        } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
// 			                            xWeight = -sx3/(6.0*_dkx3) + 2.0*sx2/_dkx2 - 8.0*sx/_dkx + 32.0/3.0;
// 			                        }
// 			                    } else if (ixModel==_nxModel-3){
// 			                        if (sx>=0.0 && sx<_dkx){
// 			                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
// 			                        } else if(sx>=_dkx && sx<(2.0*_dkx)) {
// 			                            xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
// 			                        } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
// 			                            xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
// 			                            xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
// 			                        }
// 			                    } else if (ixModel==_nxModel-2){
// 			                        if (sx>=0.0 && sx<_dkx){
// 			                            xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
// 			                        } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
// 			                            xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
// 			                            xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
// 			                        }
// 			                    } else if (ixModel==_nxModel-1){
// 			                        if (sx>=0.0 && sx<=_dkx){
// 			                            xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
// 			                        }
// 			                    }
// 							////////////////////////////////////////////////////////
//
// 								///////////// Compute weight in z-direction ////////////
//                     			for (int izModel=0; izModel<_nzModel; izModel++){
//
//                         			float sz=uValue-(*_zKnots->_mat)[izModel];
// 			                        float sz3=sz*sz*sz;
// 			                        float sz2=sz*sz;
// 			                        float zWeight=0.0;
//
//                 			        if( sz>=0.0 && sz<4.0*_dkz ){
//
//     			                        if (izModel==0){
//     			                            if (sz>=0 && sz<_dkz){
//     			                                zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
//     			                            }
//     			                        } else if (izModel==1){
//     			                            if (sz>=0 && sz<_dkz){
//     			                                zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
//     			                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
//     			                                zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
//     			                            }
//     			                        } else if (izModel==2){
//     			                            if (sz>=0 && sz<_dkz){
//     			                                zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
//     			                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
//     			                                zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
//     			                            } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
//     			                                zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
//     			                            }
//     			                        } else if (izModel>=3 && izModel<_nzModel-3){
//     			                            if (sz>=0.0 && sz<_dkz){
//     			                                // zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
//     			                                zWeight=sz3/(6.0*_dkz3);
//     			                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
//     			                                // zWeight = -sz*sz*sz/(2.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 2.0*sz/_dkz + 2.0/3.0;
//     			                                zWeight = -sz3/(2.0*_dkz3) + 2.0*sz2/_dkz2 - 2.0*sz/_dkz + 2.0/3.0;
//     			                            } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
//     			                                // zWeight = 1/(2.0*_dkz*_dkz*_dkz)*sz*sz*sz - 4.0/(_dkz*_dkz)*sz*sz + 10.0*sz/_dkz -22.0/3.0;
//     			                                zWeight = 1/(2.0*_dkz3)*sz3 - 4.0/_dkz2*sz2 + 10.0*sz/_dkz -22.0/3.0;
//     			                            } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
//     			                                // zWeight = -sz*sz*sz/(6.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 8.0*sz/_dkz + 32.0/3.0;
//     			                                zWeight = -sz3/(6.0*_dkz3) + 2.0*sz2/_dkz2 - 8.0*sz/_dkz + 32.0/3.0;
//     			                            }
//
//     			                        } else if (izModel==_nzModel-3){
//     			                            if (sz>=0.0 && sz<_dkz){
//     			                                zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
//     			                            } else if(sz>=_dkz && sz<(2.0*_dkz)) {
//     			                                zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
//     			                            } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
//     			                                zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
//     			                                zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
//     			                            }
//     			                        } else if (izModel==_nzModel-2){
//     			                            if (sz>=0.0 && sz<_dkz){
//     			                                zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
//     			                            } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
//     			                                zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
//     			                                zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
//     			                            }
//     			                        } else if (izModel==_nzModel-1){
//     			                            if (sz>=0.0 && sz<=_dkz){
//     			                                zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
//     			                            }
//     			                        }
//                                         // Add contribution to interpolated value (data)
//                             			(*data->_mat)[iyData][ixData][izData] += yWeight*xWeight*zWeight*(*_scaleVector->_mat)[iyModel][ixModel][izModel]*(*model->_mat)[iyModel][ixModel][izModel];
//                                     }
// 								}
// 							}
// 						}
// 	                }
//                 }
//             }
//         }
//     }
// }

void interpBSpline3d::forward(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const {

    // Forward: Coarse grid to fine grid
    // Model can be on an irregular grid
	if (!add) data->scale(0.0);

    // Loop over data (fine sampling grid)
	#pragma omp parallel for collapse(3)
	for (int iyData=_yFat; iyData<_nyData-_yFat; iyData++){
    	for (int ixData=_xFat; ixData<_nxData-_xFat; ixData++){
        	for (int izData=_zFat; izData<_nzData-_zFat; izData++){

            	float uValue = (*_zParamVector->_mat)[izData];
            	float vValue = (*_xParamVector->_mat)[ixData];
				float wValue = (*_yParamVector->_mat)[iyData];

                for (int iyCount=0; iyCount<_yDataIndex[iyData].size(); iyCount++){
                    int iyModel = _yDataIndex[iyData][iyCount];
					float sy=wValue-(*_yKnots->_mat)[iyModel];
					float sy3=sy*sy*sy;
					float sy2=sy*sy;
					float yWeight=0.0;

                	if( sy>=0.0 && sy<4.0*_dky ){

						///////////// Compute weight in y-direction ////////////

						if (iyModel==0){
	                        if (sy>=0 && sy<_dky){
	                            yWeight=-sy*sy*sy/(_dky*_dky*_dky)+3.0*sy*sy/(_dky*_dky)-3.0*sy/_dky+1.0;
	                        }
	                    } else if (iyModel==1){
	                        if (sy>=0 && sy<_dky){
	                            yWeight=7.0*sy*sy*sy/(4.0*_dky*_dky*_dky)-9.0*sy*sy/(2.0*_dky*_dky)+3.0*sy/_dky;
	                        } else if (sy>=_dky && sy<(2.0*_dky)){
	                            yWeight=-sy*sy*sy/(4.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-3.0*sy/_dky+2.0;
	                        }
	                    } else if (iyModel==2){
	                        if (sy>=0 && sy<_dky){
	                            yWeight=-11.0*sy*sy*sy/(12.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky);
	                        } else if (sy>=_dky && sy<(2.0*_dky)){
	                            yWeight=7.0*sy*sy*sy/(12.0*_dky*_dky*_dky)-3*sy*sy/(_dky*_dky)+9.0*sy/(2.0*_dky)-3.0/2.0;
	                        } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
	                            yWeight=-sy*sy*sy/(6.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-9.0*sy/(2.0*_dky)+9.0/2.0;
	                        }
	                    } else if (iyModel>=3 && iyModel<_nyModel-3){
	                        if (sy>=0.0 && sy<_dky){
	                            yWeight=sy3/(6.0*_dky3);
	                        } else if (sy>=_dky && sy<(2.0*_dky)){
	                            yWeight = -sy3/(2.0*_dky3) + 2.0*sy2/(_dky2) - 2.0*sy/_dky + 2.0/3.0;
	                        } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
	                            yWeight = 1/(2.0*_dky3)*sy3 - 4.0/_dky2*sy2 + 10.0*sy/_dky -22.0/3.0;
	                        } else if (sy>=(3.0*_dky) && sy<(4.0*_dky)){
	                            yWeight = -sy3/(6.0*_dky3) + 2.0*sy2/_dky2 - 8.0*sy/_dky + 32.0/3.0;
	                        }
	                    } else if (iyModel==_nyModel-3){
	                        if (sy>=0.0 && sy<_dky){
	                            yWeight=sy*sy*sy/(6.0*_dky*_dky*_dky);
	                        } else if(sy>=_dky && sy<(2.0*_dky)) {
	                            yWeight=-sy*sy*sy/(3.0*_dky*_dky*_dky)+sy*sy/(_dky*_dky)-sy/(2*_dky)+(3.0/2.0-sy/(2.0*_dky))*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky);
	                        } else if(sy>=(2.0*_dky) && sy<=(3.0*_dky)) {
	                            yWeight=sy/(3.0*_dky)*(sy*sy/(2.0*_dky*_dky)-3*sy/_dky+9.0/2.0);
	                            yWeight+=(3.0/2.0-sy/(2.0*_dky))*(-3*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky)+4*(sy-_dky)/_dky-2.0);
	                        }
	                    } else if (iyModel==_nyModel-2){
	                        if (sy>=0.0 && sy<_dky){
	                            yWeight=sy*sy*sy/(4.0*_dky*_dky*_dky);
	                        } else if(sy>=_dky && sy<=(2.0*_dky)) {
	                            yWeight=sy/(2.0*_dky)*(-3.0*sy*sy/(2.0*_dky*_dky)+4.0*sy/_dky-2.0);
	                            yWeight+=(2.0-sy/_dky)*(sy-_dky)*(sy-_dky)/(_dky*_dky);
	                        }
	                    } else if (iyModel==_nyModel-1){
	                        if (sy>=0.0 && sy<=_dky){
	                            yWeight=sy*sy*sy/(_dky*_dky*_dky);
	                        }
	                    }
						////////////////////////////////////////////////////////

						///////////// Compute weight in x-direction ////////////
                        for (int ixCount=0; ixCount<_xDataIndex[ixData].size(); ixCount++){
                            int ixModel=_xDataIndex[ixData][ixCount];
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
							////////////////////////////////////////////////////////

								///////////// Compute weight in z-direction ////////////
                                for (int izCount=0; izCount<_zDataIndex[izData].size(); izCount++){
                                    int izModel=_zDataIndex[izData][izCount];
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
                            			(*data->_mat)[iyData][ixData][izData] += yWeight*xWeight*zWeight*(*_scaleVector->_mat)[iyModel][ixModel][izModel]*(*model->_mat)[iyModel][ixModel][izModel];
                                    }
								}
							}
						}
	                }
                }
            }
        }
    }
}

// Adjoint
// void interpBSpline3d::adjoint(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const {
//
//     // Forward: Coarse grid to fine grid
//     // Model can be on an irregular grid
// 	if (!add) model->scale(0.0);
//
//     // Loop over data (fine sampling grid)
// 	#pragma omp parallel for collapse(3)
//     for (int iyModel=0; iyModel<_nyModel; iyModel++){
//         for (int ixModel=0; ixModel<_nxModel; ixModel++){
//             for (int izModel=0; izModel<_nzModel; izModel++){
//
//                 for (int iyData=_yFat; iyData<_nyData-_yFat; iyData++){
//                     float wValue = (*_yParamVector->_mat)[iyData];
//                     float sy=wValue-(*_yKnots->_mat)[iyModel];
//                     float sy3=sy*sy*sy;
//                     float sy2=sy*sy;
//                     float yWeight=0.0;
//
//                     if( sy>=0.0 && sy<4.0*_dky ){
//
//                     	///////////// Compute weight in y-direction ////////////
//
//                     	if (iyModel==0){
//                             if (sy>=0 && sy<_dky){
//                                 yWeight=-sy*sy*sy/(_dky*_dky*_dky)+3.0*sy*sy/(_dky*_dky)-3.0*sy/_dky+1.0;
//                             }
//                         } else if (iyModel==1){
//                             if (sy>=0 && sy<_dky){
//                                 yWeight=7.0*sy*sy*sy/(4.0*_dky*_dky*_dky)-9.0*sy*sy/(2.0*_dky*_dky)+3.0*sy/_dky;
//                             } else if (sy>=_dky && sy<(2.0*_dky)){
//                                 yWeight=-sy*sy*sy/(4.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-3.0*sy/_dky+2.0;
//                             }
//                         } else if (iyModel==2){
//                             if (sy>=0 && sy<_dky){
//                                 yWeight=-11.0*sy*sy*sy/(12.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky);
//                             } else if (sy>=_dky && sy<(2.0*_dky)){
//                                 yWeight=7.0*sy*sy*sy/(12.0*_dky*_dky*_dky)-3*sy*sy/(_dky*_dky)+9.0*sy/(2.0*_dky)-3.0/2.0;
//                             } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
//                                 yWeight=-sy*sy*sy/(6.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-9.0*sy/(2.0*_dky)+9.0/2.0;
//                             }
//                         } else if (iyModel>=3 && iyModel<_nyModel-3){
//                             if (sy>=0.0 && sy<_dky){
//                                 yWeight=sy3/(6.0*_dky3);
//                             } else if (sy>=_dky && sy<(2.0*_dky)){
//                                 yWeight = -sy3/(2.0*_dky3) + 2.0*sy2/(_dky2) - 2.0*sy/_dky + 2.0/3.0;
//                             } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
//                                 yWeight = 1/(2.0*_dky3)*sy3 - 4.0/_dky2*sy2 + 10.0*sy/_dky -22.0/3.0;
//                             } else if (sy>=(3.0*_dky) && sy<(4.0*_dky)){
//                                 yWeight = -sy3/(6.0*_dky3) + 2.0*sy2/_dky2 - 8.0*sy/_dky + 32.0/3.0;
//                             }
//                         } else if (iyModel==_nyModel-3){
//                             if (sy>=0.0 && sy<_dky){
//                                 yWeight=sy*sy*sy/(6.0*_dky*_dky*_dky);
//                             } else if(sy>=_dky && sy<(2.0*_dky)) {
//                                 yWeight=-sy*sy*sy/(3.0*_dky*_dky*_dky)+sy*sy/(_dky*_dky)-sy/(2*_dky)+(3.0/2.0-sy/(2.0*_dky))*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky);
//                             } else if(sy>=(2.0*_dky) && sy<=(3.0*_dky)) {
//                                 yWeight=sy/(3.0*_dky)*(sy*sy/(2.0*_dky*_dky)-3*sy/_dky+9.0/2.0);
//                                 yWeight+=(3.0/2.0-sy/(2.0*_dky))*(-3*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky)+4*(sy-_dky)/_dky-2.0);
//                             }
//                         } else if (iyModel==_nyModel-2){
//                             if (sy>=0.0 && sy<_dky){
//                                 yWeight=sy*sy*sy/(4.0*_dky*_dky*_dky);
//                             } else if(sy>=_dky && sy<=(2.0*_dky)) {
//                                 yWeight=sy/(2.0*_dky)*(-3.0*sy*sy/(2.0*_dky*_dky)+4.0*sy/_dky-2.0);
//                                 yWeight+=(2.0-sy/_dky)*(sy-_dky)*(sy-_dky)/(_dky*_dky);
//                             }
//                         } else if (iyModel==_nyModel-1){
//                             if (sy>=0.0 && sy<=_dky){
//                                 yWeight=sy*sy*sy/(_dky*_dky*_dky);
//                             }
//                         }
//                     	////////////////////////////////////////////////////////
//
//                     	///////////// Compute weight in x-direction ////////////
//                     	for (int ixData=_xFat; ixData<_nxData-_xFat; ixData++){
//                             float vValue = (*_xParamVector->_mat)[ixData];
//                             float sx=vValue-(*_xKnots->_mat)[ixModel];
//                             float sx3=sx*sx*sx;
//                             float sx2=sx*sx;
//                             float xWeight=0.0;
//
//                     		if( sx>=0.0 && sx<4.0*_dkx ){
//
//                                 if (ixModel==0){
//                                     if (sx>=0 && sx<_dkx){
//                                         xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
//                                     }
//                                 } else if (ixModel==1){
//                                     if (sx>=0 && sx<_dkx){
//                                         xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
//                                     } else if (sx>=_dkx && sx<(2.0*_dkx)){
//                                         xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
//                                     }
//                                 } else if (ixModel==2){
//                                     if (sx>=0 && sx<_dkx){
//                                         xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
//                                     } else if (sx>=_dkx && sx<(2.0*_dkx)){
//                                         xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
//                                     } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
//                                         xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
//                                     }
//                                 } else if (ixModel>=3 && ixModel<_nxModel-3){
//                                     if (sx>=0.0 && sx<_dkx){
//                                         xWeight=sx3/(6.0*_dkx3);
//                                     } else if (sx>=_dkx && sx<(2.0*_dkx)){
//                                         xWeight = -sx3/(2.0*_dkx3) + 2.0*sx2/(_dkx2) - 2.0*sx/_dkx + 2.0/3.0;
//                                     } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
//                                         xWeight = 1/(2.0*_dkx3)*sx3 - 4.0/_dkx2*sx2 + 10.0*sx/_dkx -22.0/3.0;
//                                     } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
//                                         xWeight = -sx3/(6.0*_dkx3) + 2.0*sx2/_dkx2 - 8.0*sx/_dkx + 32.0/3.0;
//                                     }
//                                 } else if (ixModel==_nxModel-3){
//                                     if (sx>=0.0 && sx<_dkx){
//                                         xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
//                                     } else if(sx>=_dkx && sx<(2.0*_dkx)) {
//                                         xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
//                                     } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
//                                         xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
//                                         xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
//                                     }
//                                 } else if (ixModel==_nxModel-2){
//                                     if (sx>=0.0 && sx<_dkx){
//                                         xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
//                                     } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
//                                         xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
//                                         xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
//                                     }
//                                 } else if (ixModel==_nxModel-1){
//                                     if (sx>=0.0 && sx<=_dkx){
//                                         xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
//                                     }
//                                 }
//
//
//                     		    /////////// Compute weight in z-direction ////////////
//                     		    for (int izData=_zFat; izData<_nzData-_zFat; izData++){
//                                     float uValue = (*_zParamVector->_mat)[izData];
//                             		float sz=uValue-(*_zKnots->_mat)[izModel];
//                                     float sz3=sz*sz*sz;
//                                     float sz2=sz*sz;
//                                     float zWeight=0.0;
//
//                     			    if( sz>=0.0 && sz<4.0*_dkz ){
//
//                                         if (izModel==0){
//                                             if (sz>=0 && sz<_dkz){
//                                                 zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
//                                             }
//                                         } else if (izModel==1){
//                                             if (sz>=0 && sz<_dkz){
//                                                 zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
//                                             } else if (sz>=_dkz && sz<(2.0*_dkz)){
//                                                 zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
//                                             }
//                                         } else if (izModel==2){
//                                             if (sz>=0 && sz<_dkz){
//                                                 zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
//                                             } else if (sz>=_dkz && sz<(2.0*_dkz)){
//                                                 zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
//                                             } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
//                                                 zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
//                                             }
//                                         } else if (izModel>=3 && izModel<_nzModel-3){
//                                             if (sz>=0.0 && sz<_dkz){
//                                                 // zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
//                                                 zWeight=sz3/(6.0*_dkz3);
//                                             } else if (sz>=_dkz && sz<(2.0*_dkz)){
//                                                 // zWeight = -sz*sz*sz/(2.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 2.0*sz/_dkz + 2.0/3.0;
//                                                 zWeight = -sz3/(2.0*_dkz3) + 2.0*sz2/_dkz2 - 2.0*sz/_dkz + 2.0/3.0;
//                                             } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
//                                                 // zWeight = 1/(2.0*_dkz*_dkz*_dkz)*sz*sz*sz - 4.0/(_dkz*_dkz)*sz*sz + 10.0*sz/_dkz -22.0/3.0;
//                                                 zWeight = 1/(2.0*_dkz3)*sz3 - 4.0/_dkz2*sz2 + 10.0*sz/_dkz -22.0/3.0;
//                                             } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
//                                                 // zWeight = -sz*sz*sz/(6.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 8.0*sz/_dkz + 32.0/3.0;
//                                                 zWeight = -sz3/(6.0*_dkz3) + 2.0*sz2/_dkz2 - 8.0*sz/_dkz + 32.0/3.0;
//                                             }
//
//                                         } else if (izModel==_nzModel-3){
//                                             if (sz>=0.0 && sz<_dkz){
//                                                 zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
//                                             } else if(sz>=_dkz && sz<(2.0*_dkz)) {
//                                                 zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
//                                             } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
//                                                 zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
//                                                 zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
//                                             }
//                                         } else if (izModel==_nzModel-2){
//                                             if (sz>=0.0 && sz<_dkz){
//                                                 zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
//                                             } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
//                                                 zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
//                                                 zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
//                                             }
//                                         } else if (izModel==_nzModel-1){
//                                             if (sz>=0.0 && sz<=_dkz){
//                                                 zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
//                                             }
//                                         }
//
//                                 		// Add contribution to interpolated value (data)
//                                 		(*model->_mat)[iyModel][ixModel][izModel] += yWeight*xWeight*zWeight*(*_scaleVector->_mat)[iyModel][ixModel][izModel]*(*data->_mat)[iyData][ixData][izData];
//                     				}
//                     			}
//                     		}
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

void interpBSpline3d::adjoint(const bool add, const std::shared_ptr<float3DReg> model, std::shared_ptr<float3DReg> data) const {

    // Forward: Coarse grid to fine grid
    // Model can be on an irregular grid
	if (!add) model->scale(0.0);

    // Loop over data (fine sampling grid)
	#pragma omp parallel for collapse(3)
    for (int iyModel=0; iyModel<_nyModel; iyModel++){
        for (int ixModel=0; ixModel<_nxModel; ixModel++){
            for (int izModel=0; izModel<_nzModel; izModel++){

                for (int iyCount=0; iyCount<_yModelIndex[iyModel].size(); iyCount++){
                    int iyData = _yModelIndex[iyModel][iyCount];
                    float wValue = (*_yParamVector->_mat)[iyData];
                    float sy=wValue-(*_yKnots->_mat)[iyModel];
                    float sy3=sy*sy*sy;
                    float sy2=sy*sy;
                    float yWeight=0.0;

                    if( sy>=0.0 && sy<4.0*_dky ){

                    	///////////// Compute weight in y-direction ////////////

                    	if (iyModel==0){
                            if (sy>=0 && sy<_dky){
                                yWeight=-sy*sy*sy/(_dky*_dky*_dky)+3.0*sy*sy/(_dky*_dky)-3.0*sy/_dky+1.0;
                            }
                        } else if (iyModel==1){
                            if (sy>=0 && sy<_dky){
                                yWeight=7.0*sy*sy*sy/(4.0*_dky*_dky*_dky)-9.0*sy*sy/(2.0*_dky*_dky)+3.0*sy/_dky;
                            } else if (sy>=_dky && sy<(2.0*_dky)){
                                yWeight=-sy*sy*sy/(4.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-3.0*sy/_dky+2.0;
                            }
                        } else if (iyModel==2){
                            if (sy>=0 && sy<_dky){
                                yWeight=-11.0*sy*sy*sy/(12.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky);
                            } else if (sy>=_dky && sy<(2.0*_dky)){
                                yWeight=7.0*sy*sy*sy/(12.0*_dky*_dky*_dky)-3*sy*sy/(_dky*_dky)+9.0*sy/(2.0*_dky)-3.0/2.0;
                            } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
                                yWeight=-sy*sy*sy/(6.0*_dky*_dky*_dky)+3.0*sy*sy/(2.0*_dky*_dky)-9.0*sy/(2.0*_dky)+9.0/2.0;
                            }
                        } else if (iyModel>=3 && iyModel<_nyModel-3){
                            if (sy>=0.0 && sy<_dky){
                                yWeight=sy3/(6.0*_dky3);
                            } else if (sy>=_dky && sy<(2.0*_dky)){
                                yWeight = -sy3/(2.0*_dky3) + 2.0*sy2/(_dky2) - 2.0*sy/_dky + 2.0/3.0;
                            } else if (sy>=(2.0*_dky) && sy<(3.0*_dky)){
                                yWeight = 1/(2.0*_dky3)*sy3 - 4.0/_dky2*sy2 + 10.0*sy/_dky -22.0/3.0;
                            } else if (sy>=(3.0*_dky) && sy<(4.0*_dky)){
                                yWeight = -sy3/(6.0*_dky3) + 2.0*sy2/_dky2 - 8.0*sy/_dky + 32.0/3.0;
                            }
                        } else if (iyModel==_nyModel-3){
                            if (sy>=0.0 && sy<_dky){
                                yWeight=sy*sy*sy/(6.0*_dky*_dky*_dky);
                            } else if(sy>=_dky && sy<(2.0*_dky)) {
                                yWeight=-sy*sy*sy/(3.0*_dky*_dky*_dky)+sy*sy/(_dky*_dky)-sy/(2*_dky)+(3.0/2.0-sy/(2.0*_dky))*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky);
                            } else if(sy>=(2.0*_dky) && sy<=(3.0*_dky)) {
                                yWeight=sy/(3.0*_dky)*(sy*sy/(2.0*_dky*_dky)-3*sy/_dky+9.0/2.0);
                                yWeight+=(3.0/2.0-sy/(2.0*_dky))*(-3*(sy-_dky)*(sy-_dky)/(2.0*_dky*_dky)+4*(sy-_dky)/_dky-2.0);
                            }
                        } else if (iyModel==_nyModel-2){
                            if (sy>=0.0 && sy<_dky){
                                yWeight=sy*sy*sy/(4.0*_dky*_dky*_dky);
                            } else if(sy>=_dky && sy<=(2.0*_dky)) {
                                yWeight=sy/(2.0*_dky)*(-3.0*sy*sy/(2.0*_dky*_dky)+4.0*sy/_dky-2.0);
                                yWeight+=(2.0-sy/_dky)*(sy-_dky)*(sy-_dky)/(_dky*_dky);
                            }
                        } else if (iyModel==_nyModel-1){
                            if (sy>=0.0 && sy<=_dky){
                                yWeight=sy*sy*sy/(_dky*_dky*_dky);
                            }
                        }
                    	////////////////////////////////////////////////////////

                    	///////////// Compute weight in x-direction ////////////
                        for (int ixCount=0; ixCount<_xModelIndex[ixModel].size(); ixCount++){
                            int ixData = _xModelIndex[ixModel][ixCount];
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


                    		    /////////// Compute weight in z-direction ////////////
                                for (int izCount=0; izCount<_zModelIndex[izModel].size(); izCount++){
                                    int izData = _zModelIndex[izModel][izCount];
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
                                		(*model->_mat)[iyModel][ixModel][izModel] += yWeight*xWeight*zWeight*(*_scaleVector->_mat)[iyModel][ixModel][izModel]*(*data->_mat)[iyData][ixData][izData];
                    				}
                    			}
                    		}
                        }
                    }
                }
            }
        }
    }
}

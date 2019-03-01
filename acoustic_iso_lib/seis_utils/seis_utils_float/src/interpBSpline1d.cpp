#include <string>
#include <float2DReg.h>
#include <iostream>
#include "interpBSpline1d.h"
#include <omp.h>
#include <vector>

// Contructor
interpBSpline1d::interpBSpline1d(int zOrder, std::shared_ptr<float1DReg> zModel, axis zDataAxis, int nzParamVector, int scaling, float zTolerance, int fat){

    // B-spline parameters
    _zOrder = zOrder;
    _scaling = scaling;

    // Model
    _zModel=zModel;
    _nzModel = _zModel->getHyper()->getAxis(1).n;
    _nModel = _nzModel;
    _scaleVector = std::make_shared<float1DReg>(_nzModel);
    for (int izModel=0; izModel<_nzModel; izModel++){
        (*_scaleVector->_mat)[izModel]=1.0;
    }

    // Data
    _fat = fat;
    _zDataAxis = zDataAxis; // z-coordinates of data points assumed to be uniformly distributed
    _nzData = _zDataAxis.n;
    _nData =  _nzData;
    _zData = std::make_shared<float1DReg>(_nData);
    for (int izData=0; izData<_nzData; izData++){
        (*_zData->_mat)[izData]=_zDataAxis.o+izData*_zDataAxis.d;
    }

    // Set the tolerance [km]
    _zTolerance=zTolerance*_zDataAxis.d;

    // Number of points to evaluate in the parameter vectors
    _nzParamVector = nzParamVector;

    // Build the knot vectors
    buildKnotVector();

    // Compute parameter vectors
    _zParamVector = computeParamVectorZ();

    // Compute scale for amplitude balancing
    computeScaleVector();

}

// Knot vectors for both directions
void interpBSpline1d::buildKnotVector() {

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

}

// Compute parameter vector containing optimal parameters
std::shared_ptr<float1DReg> interpBSpline1d::computeParamVectorZ(){

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
                zInterp+=zWeight*(*_zModel->_mat)[izModel];

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

// Scaling vector
void interpBSpline1d::computeScaleVector(){

    // Variables declaration
    float uValue, zWeight;
	std::shared_ptr<SEP::float1DReg> scaleVectorData(new SEP::float1DReg(_nzData));
    std::shared_ptr<SEP::float1DReg> scaleVectorTemp(new SEP::float1DReg(_nzModel));
    scaleVectorData->scale(0.0);
    scaleVectorTemp->scale(0.0);

    if (_scaling == 1){

        // Apply one forward
        forward(false, _scaleVector, scaleVectorData);

        // Apply one adjoint
        adjoint(false, scaleVectorTemp, scaleVectorData);

        // Compute scaling
        for (int izModel=0; izModel<_nzModel; izModel++){
            (*_scaleVector->_mat)[izModel]=1.0/sqrt((*scaleVectorTemp->_mat)[izModel]);
        }
    }
}

// Forward
void interpBSpline1d::forward(const bool add, const std::shared_ptr<float1DReg> model, std::shared_ptr<float1DReg> data) const {

    // Forward: Coarse grid to fine grid
    // Model can be on an irregular grid
	if (!add) data->scale(0.0);

    #pragma omp parallel for
    for (int izData=_fat; izData<_nzData-_fat; izData++){
        float uValue = (*_zParamVector->_mat)[izData];

        for (int izModel=0; izModel<_nzModel; izModel++){

            float sz=uValue-(*_zKnots->_mat)[izModel];
            float sz3=sz*sz*sz;
            float sz2=sz*sz;
            float zWeight=0.0;

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
            (*data->_mat)[izData] += zWeight*(*_scaleVector->_mat)[izModel]*(*model->_mat)[izModel];
        }
    }
}

// Adjoint
void interpBSpline1d::adjoint(const bool add, std::shared_ptr<float1DReg> model, const std::shared_ptr<float1DReg> data) const {

    // Adjoint: Fine grid to coarse grid
    // Model can be on an irregular grid
    if (!add) model->scale(0.0);

    // #pragma omp parallel for
    for (int izModel=0; izModel<_nzModel; izModel++){

        for (int izData=_fat; izData<_nzData-_fat; izData++){
            float uValue = (*_zParamVector->_mat)[izData];
            float sz=uValue-(*_zKnots->_mat)[izModel];
            float sz3=sz*sz*sz;
            float sz2=sz*sz;
            float zWeight=0.0;

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
            (*model->_mat)[izModel] += zWeight*(*_scaleVector->_mat)[izModel]*(*data->_mat)[izData];
        }
    }
}

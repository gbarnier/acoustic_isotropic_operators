#include <string>
#include <double2DReg.h>
#include <iostream>
#include "interpBSpline2dDouble.h"
#include <omp.h>
#include <vector>

// Contructor
interpBSpline2dDouble::interpBSpline2dDouble(int zOrder, int xOrder, std::shared_ptr<double1DReg> zControlPoints, std::shared_ptr<double1DReg> xControlPoints, axis zDataAxis, axis xDataAxis, int nzParamVector, int nxParamVector, int scaling, double zTolerance, double xTolerance, int fat){

    // B-spline parameters
    _zOrder = zOrder; // Order of interpolation in the z-direction
    _xOrder = xOrder; // Order of interpolation in the x-direction
    _scaling = scaling; // if = 1, compute and apply scaling to balance operator amplitudes

    // Model
    _zControlPoints=zControlPoints;
    _xControlPoints=xControlPoints;
    _nzModel = _zControlPoints->getHyper()->getAxis(1).n; // Number of control points in the z-direction
    _nxModel = _xControlPoints->getHyper()->getAxis(1).n; // Number of control points in the x-direction
    _nModel = _nzModel*_nxModel; // Total model size
    _zModel = std::make_shared<double2DReg>(_nzModel, _nxModel);
    _xModel = std::make_shared<double2DReg>(_nzModel, _nxModel);

    // Create 2D mesh for z and x
    for (int ix=0; ix<_nxModel; ix++){
        double x = (*_xControlPoints->_mat)[ix];
        for (int iz=0; iz<_nzModel; iz++){
            double z = (*_zControlPoints->_mat)[iz];
            (*_zModel->_mat)[ix][iz] = z;
            (*_xModel->_mat)[ix][iz] = x;
        }
    }

    // Data
    _fat = fat;
    _zDataAxis = zDataAxis; // z-coordinates of data points assumed to be uniformly distributed
    _xDataAxis = xDataAxis; // x-coordinates of data points assumed to be uniformly distributed
    _nzData = _zDataAxis.n;
    _nxData = _xDataAxis.n;
    _nData =  _zDataAxis.n*_xDataAxis.n;
    _zData = std::make_shared<double2DReg>(_zDataAxis, _xDataAxis);
    _xData = std::make_shared<double2DReg>(_zDataAxis, _xDataAxis);

    // Fill in data grid in z and x
	#pragma omp parallel for collapse(2)
    for (int ixData=0; ixData<_xDataAxis.n; ixData++){
        for (int izData=0; izData <_zDataAxis.n; izData++){
            double z = _zDataAxis.o+izData*_zDataAxis.d; // z-position
            double x = _xDataAxis.o+ixData*_xDataAxis.d; // x-position
            (*_zData->_mat)[ixData][izData] = z; // z-position
            (*_xData->_mat)[ixData][izData] = x; // x-position
        }
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
    std::cout << "Done zParameter" << std::endl;
    _xParamVector = computeParamVectorX();
    std::cout << "Done xParameter" << std::endl;

    // Compute scale for amplitude balancing
    // _scaleVector = computeScaleVector(_scaling);

}

interpBSpline2dDouble::interpBSpline2dDouble(int zOrder, int xOrder, std::shared_ptr<double1DReg> zControlPoints, std::shared_ptr<double1DReg> xControlPoints, axis zDataAxis, axis xDataAxis, std::shared_ptr<double1DReg> zParamVector, std::shared_ptr<double1DReg> xParamVector, int scaling, double zTolerance, double xTolerance, int fat){

    // B-spline parameters
    _zOrder = zOrder; // Order of interpolation in the z-direction
    _xOrder = xOrder; // Order of interpolation in the x-direction
    _scaling = scaling; // if = 1, compute and apply scaling to balance operator amplitudes

    // Model
    _zControlPoints=zControlPoints;
    _xControlPoints=xControlPoints;
    _nzModel = _zControlPoints->getHyper()->getAxis(1).n; // Number of control points in the z-direction
    _nxModel = _xControlPoints->getHyper()->getAxis(1).n; // Number of control points in the x-direction
    _nModel = _nzModel*_nxModel; // Total model size
    _zModel = std::make_shared<double2DReg>(_nzModel, _nxModel);
    _xModel = std::make_shared<double2DReg>(_nzModel, _nxModel);

    // Create 2D mesh for z and x
    for (int ix=0; ix<_nxModel; ix++){
        double x = (*_xControlPoints->_mat)[ix];
        for (int iz=0; iz<_nzModel; iz++){
            double z = (*_zControlPoints->_mat)[iz];
            (*_zModel->_mat)[ix][iz] = z;
            (*_xModel->_mat)[ix][iz] = x;
        }
    }

    // Data
    _fat = fat;
    _zDataAxis = zDataAxis; // z-coordinates of data points assumed to be uniformly distributed
    _xDataAxis = xDataAxis; // x-coordinates of data points assumed to be uniformly distributed
    _nzData = _zDataAxis.n;
    _nxData = _xDataAxis.n;
    _nData =  _zDataAxis.n*_xDataAxis.n;
    _zData = std::make_shared<double2DReg>(_zDataAxis, _xDataAxis);
    _xData = std::make_shared<double2DReg>(_zDataAxis, _xDataAxis);

    // Fill in data grid in z and x
	#pragma omp parallel for collapse(2)
    for (int ixData=0; ixData<_xDataAxis.n; ixData++){
        for (int izData=0; izData <_zDataAxis.n; izData++){
            double z = _zDataAxis.o+izData*_zDataAxis.d; // z-position
            double x = _xDataAxis.o+ixData*_xDataAxis.d; // x-position
            (*_zData->_mat)[ixData][izData] = z; // z-position
            (*_xData->_mat)[ixData][izData] = x; // x-position
        }
    }

    // Set the tolerance [km]
    _zTolerance=zTolerance*_zDataAxis.d;
    _xTolerance=xTolerance*_xDataAxis.d;

    // Build the knot vectors
    buildKnotVectors2d();

    // Set parameter vectors
    _zParamVector = zParamVector;
    _xParamVector = xParamVector;

    // Compute scale for amplitude balancing
    _scaleVector = computeScaleVector(_scaling);

}

// Knot vectors for both directions
void interpBSpline2dDouble::buildKnotVectors2d() {

    // Knots for the z-axis
    _nkz=_nzModel+_zOrder+1; // Number of knots
    _nkzSimple=_nkz-2*_zOrder; // Number of simple knots
    _okz = 0.0; // Position of FIRST knot
    _fkz = 1.0; // Position of LAST knot
    _dkz=(_fkz-_okz)/(_nkzSimple-1); // Knot sampling
    _kzAxis = axis(_nkz, _okz, _dkz); // Knot axis
    _zKnots = std::make_shared<double1DReg>(_kzAxis);

    // Knots for the x-axis
    _nkx=_nxModel+_xOrder+1; // Number of knots
    _nkxSimple=_nkx-2*_xOrder; // Number of simple knots
    _okx = 0.0; // Position of FIRST knot
    _fkx = 1.0; // Position of LAST knot
    _dkx=(_fkx-_okx)/(_nkxSimple-1); // Knot sampling
    _kxAxis = axis(_nkx, _okx, _dkx); // Knot axis
    _xKnots = std::make_shared<double1DReg>(_kxAxis);

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

// B-Spline weights
double interpBSpline2dDouble::bspline1d(int iControl, double uValue, int order, double dk, int nControlPoints, std::shared_ptr<double1DReg> knots) {

    // Compute shift
    double s=uValue-(*knots->_mat)[iControl];
    double weight=0;

    /********************************* Order 0 ********************************/
    if (order == 0){
        if (s>=0.0 && s<dk){
            weight = 1;
        }
    }
    /********************************* Order 1 ********************************/
    else if (order == 1){
        // First control point (double knot)
        if (iControl==0) {
            if (s>=0 && s<dk){
                weight=1.0-s/dk;
            }
        }
        // Main loop (single knots)
        else if (iControl>=1 && iControl < nControlPoints-1){
            if (s>=0.0 && s<dk){
                weight=s/dk;
            } else if (s>=dk && s<(2.0*dk)){
                weight=2.0-s/dk;
            }
        }
        // Last control point (double knots)
        else if (nControlPoints-1){
            if (s>=0.0 && s<=dk){
                weight=s/dk;
            }
        }
    }
    /********************************* Order 2 ********************************/
    else if (order == 2){
        // i = 0
        if (iControl==0){
            if (s>=0.0 && s<dk){
				weight=s*s/(dk*dk)-2.0*s/dk+1.0;
			}
        // i = 1
        } else if (iControl==1){
            if (s>=0.0 && s<dk){
                weight=-3.0/2.0*s*s/(dk*dk)+2.0*s/dk;
            } else if (s>=dk && s<(2.0*dk)) {
                weight=s*s/(2.0*dk*dk)-2.0*s/dk+2;
            }
        // Main loop
        } else if (iControl>=2 && iControl<nControlPoints-2){
            if (s>=0.0 && s<dk){
                weight=s*s/(2.0*dk*dk);
            } else if (s>=dk && s<(2.0*dk)){
                weight=-s*s/(dk*dk)+3.0*s/dk-1.5;
            } else if (s>=(2.0*dk) && s<(3.0*dk)){
                weight=s*s/(2.0*dk*dk)-3.0*s/dk+4.5;
            }
        // i=nControlPoint-2
        } else if (iControl==nControlPoints-2){
            if (s>=0.0 && s<dk){
				weight=s*s/(2.0*dk*dk);
			} else if(s>=dk && s<(2.0*dk)) {
				weight=-3.0*s*s/(2.0*dk*dk)+4.0*s/dk-2.0;
			}
        // i=nControlPoint-1
        } else if (iControl==nControlPoints-1){
            if (s>=0.0 && s<dk){
				weight=s*s/(dk*dk);
			}
        }
    }
    /********************************* Order 3 ********************************/
    else if (order == 3){
        // i = 0
        if (iControl==0){
            if (s>=0 && s<dk){
                weight=-s*s*s/(dk*dk*dk)+3.0*s*s/(dk*dk)-3.0*s/dk+1.0;
            }
        // i = 1
        } else if (iControl==1){
            if (s>=0 && s<dk){
				weight=7.0*s*s*s/(4.0*dk*dk*dk)-9.0*s*s/(2.0*dk*dk)+3.0*s/dk;
			} else if (s>=dk && s<(2.0*dk)){
				weight=-s*s*s/(4.0*dk*dk*dk)+3.0*s*s/(2.0*dk*dk)-3.0*s/dk+2.0;
			}
        // i = 2
        } else if (iControl==2){
            if (s>=0 && s<dk){
				weight=-11.0*s*s*s/(12.0*dk*dk*dk)+3.0*s*s/(2.0*dk*dk);
			} else if (s>=dk && s<(2.0*dk)){
				weight=7.0*s*s*s/(12.0*dk*dk*dk)-3*s*s/(dk*dk)+9.0*s/(2.0*dk)-3.0/2.0;
			} else if (s>=(2.0*dk) && s<(3.0*dk)){
				weight=-s*s*s/(6.0*dk*dk*dk)+3.0*s*s/(2.0*dk*dk)-9.0*s/(2.0*dk)+9.0/2.0;
			}
        // Main loop
        } else if (iControl>=3 && iControl<nControlPoints-3){
            if (s>=0.0 && s<dk){
                weight=s*s*s/(6.0*dk*dk*dk);
            } else if (s>=dk && s<(2.0*dk)){
                weight = -s*s*s/(2.0*dk*dk*dk) + 2.0*s*s/(dk*dk) - 2.0*s/dk + 2.0/3.0;
            } else if (s>=(2.0*dk) && s<(3.0*dk)){
                weight = 1/(2.0*dk*dk*dk)*s*s*s - 4.0/(dk*dk)*s*s + 10.0*s/dk -22.0/3.0;
            } else if (s>=(3.0*dk) && s<(4.0*dk)){
                weight = -s*s*s/(6.0*dk*dk*dk) + 2.0*s*s/(dk*dk) - 8.0*s/dk + 32.0/3.0;
            }
        // i=nControlPoints-3
        } else if (iControl==nControlPoints-3){
            if (s>=0.0 && s<dk){
				weight=s*s*s/(6.0*dk*dk*dk);
			} else if(s>=dk && s<(2.0*dk)) {
				weight=-s*s*s/(3.0*dk*dk*dk)+s*s/(dk*dk)-s/(2*dk)+(3.0/2.0-s/(2.0*dk))*(s-dk)*(s-dk)/(2.0*dk*dk);
			} else if(s>=(2.0*dk) && s<=(3.0*dk)) {
				weight=s/(3.0*dk)*(s*s/(2.0*dk*dk)-3*s/dk+9.0/2.0);
				weight+=(3.0/2.0-s/(2.0*dk))*(-3*(s-dk)*(s-dk)/(2.0*dk*dk)+4*(s-dk)/dk-2.0);
			}
        // i=nControlPoints-2
        } else if (iControl==nControlPoints-2){
            if (s>=0.0 && s<dk){
				weight=s*s*s/(4.0*dk*dk*dk);
			} else if(s>=dk && s<=(2.0*dk)) {
				weight=s/(2.0*dk)*(-3.0*s*s/(2.0*dk*dk)+4.0*s/dk-2.0);
				weight+=(2.0-s/dk)*(s-dk)*(s-dk)/(dk*dk);
			}
        // i=nControlPoints-1
        } else if (iControl==nControlPoints-1){
            if (s>=0.0 && s<=dk){
				weight=s*s*s/(dk*dk*dk);
			}
        }
    }
    /********************************* Order error ****************************/
    else {
        std::cout << "**** ERROR: Order not supported. Please choose between {0,1,2,3} ****" << std::endl;
        assert(1==2);
    }
    return weight;
}

// Compute parameter vector containing optimal parameters
std::shared_ptr<double1DReg> interpBSpline2dDouble::computeParamVectorZ(){

    // Generate u (z-direction)
    int nu = _nzParamVector;
    double ou = _okz;
	double fu = _fkz;
	double du = (fu-ou)/(nu-1);
	nu=nu-1;

    std::shared_ptr<double1DReg> u(new double1DReg(nu));
    axis uAxis = axis(nu, ou, du);
    std::shared_ptr<double1DReg> paramVector(new double1DReg(_nzData));

    // Initialize param vector with -1
    #pragma omp parallel for
    for (int iData=0; iData<_nzData; iData++){
        (*paramVector->_mat)[iData]=-1.0;
    }

    // Loop over data space
	#pragma omp parallel for
    for (int izData=_fat; izData<_nzData-_fat; izData++){

        double error=100000;
        for (int iu=0; iu<nu; iu++){

            double uValue=ou+iu*du;
            double zInterp = 0;
            for (int izModel=0; izModel<_nzModel; izModel++){

                double sz=uValue-(*_zKnots->_mat)[izModel];
                double zWeight=0.0;

                //////////// Compute interpolation weight //////////////////
                // Order 0
                if (_zOrder==0){
                    if (sz>=0.0 && sz<_dkz){
                        zWeight = 1.0;
                    }
                }
                // Order 1
                else if (_zOrder==1){
                    if (izModel==0) {
                        if (sz>=0 && sz<_dkz){
                            zWeight=1.0-sz/_dkz;
                        }
                    }
                    else if (izModel>=1 && izModel < _nzModel-1){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz/_dkz;
                        } else if (sz>=_dkz && sz<(2.0*_dkz)){
                            zWeight=2.0-sz/_dkz;
                        }
                    }
                    else if (_nzModel-1){
                        if (sz>=0.0 && sz<=_dkz){
                            zWeight=sz/_dkz;
                        }
                    }
                }
                // Order 2
                else if (_zOrder==2){
                    // i = 0
                    if (izModel==0){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz*sz/(_dkz*_dkz)-2.0*sz/_dkz+1.0;
                        }
                    // i = 1
                    } else if (izModel==1){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=-3.0/2.0*sz*sz/(_dkz*_dkz)+2.0*sz/_dkz;
                        } else if (sz>=_dkz && sz<(2.0*_dkz)) {
                            zWeight=sz*sz/(2.0*_dkz*_dkz)-2.0*sz/_dkz+2;
                        }
                    // Main loop
                    } else if (izModel>=2 && izModel<_nzModel-2){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz*sz/(2.0*_dkz*_dkz);
                        } else if (sz>=_dkz && sz<(2.0*_dkz)){
                            zWeight=-sz*sz/(_dkz*_dkz)+3.0*sz/_dkz-1.5;
                        } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                            zWeight=sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+4.5;
                        }
                    // i=nControlPoint-2
                    } else if (izModel==_nzModel-2){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz*sz/(2.0*_dkz*_dkz);
                        } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                            zWeight=-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0;
                        }
                    // i=nControlPoint-1
                    } else if (izModel==_nzModel-1){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz*sz/(_dkz*_dkz);
                        }
                    }
                }
                // Order 3
                else if (_zOrder==3){
                    // i = 0
                    if (izModel==0){
                        if (sz>=0 && sz<_dkz){
                            zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
                        }
                    // i = 1
                    } else if (izModel==1){
                        if (sz>=0 && sz<_dkz){
                            zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
                        } else if (sz>=_dkz && sz<(2.0*_dkz)){
                            zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
                        }
                    // i = 2
                    } else if (izModel==2){
                        if (sz>=0 && sz<_dkz){
                            zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
                        } else if (sz>=_dkz && sz<(2.0*_dkz)){
                            zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
                        } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                            zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
                        }
                    // Main loop
                    } else if (izModel>=3 && izModel<_nzModel-3){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                        } else if (sz>=_dkz && sz<(2.0*_dkz)){
                            zWeight = -sz*sz*sz/(2.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 2.0*sz/_dkz + 2.0/3.0;
                        } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                            zWeight = 1/(2.0*_dkz*_dkz*_dkz)*sz*sz*sz - 4.0/(_dkz*_dkz)*sz*sz + 10.0*sz/_dkz -22.0/3.0;
                        } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
                            zWeight = -sz*sz*sz/(6.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 8.0*sz/_dkz + 32.0/3.0;
                        }
                    // i=_nzModel-3
                    } else if (izModel==_nzModel-3){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                        } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                            zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
                        } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
                            zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
                            zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
                        }
                    // i=_nzModel-2
                    } else if (izModel==_nzModel-2){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
                        } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
                            zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
                            zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
                        }
                    // i=_nzModel-1
                    } else if (izModel==_nzModel-1){
                        if (sz>=0.0 && sz<=_dkz){
                            zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
                        }
                    }
                }
                ////////////////////////////////////////////////////////////

                // Add contribution of model point
                zInterp+=zWeight*(*_zModel->_mat)[0][izModel];

            }
            // Finished computing interpolated position for this u-value
            // Update the optimal u-value if interpolated point is clsoer to data point
            if (std::abs(zInterp-(*_zData->_mat)[0][izData]) < error) {
                error=std::abs(zInterp-(*_zData->_mat)[0][izData]);
                (*paramVector->_mat)[izData]=uValue;
            }
        }
        // Finished computing interpolated values for all u's
        if (std::abs(error)>_zTolerance){
            std::cout << "**** ERROR: Could not find a parameter for data point in the z-direction #" << izData << " " << (*_zData->_mat)[0][izData] << " [km]. Try increasing the number of samples! ****" << std::endl;
            std::cout << "Error = " << error << std::endl;
            std::cout << "Tolerance = " << _zTolerance << " [km]" << std::endl;
            assert(1==2);
        }
    }
    return paramVector;
}
std::shared_ptr<double1DReg> interpBSpline2dDouble::computeParamVectorX(){

    // Generate u (z-direction)
    int nv = _nxParamVector;
    double ov = _okx;
	double fv = _fkx;
	double dv = (fv-ov)/(nv-1);
	nv=nv-1;

    std::shared_ptr<double1DReg> v(new double1DReg(nv));
    axis vAxis = axis(nv, ov, dv);
    std::shared_ptr<double1DReg> paramVector(new double1DReg(_nxData));

    // Initialize param vector with -1
    #pragma omp parallel for
    for (int iData=0; iData<_nxData; iData++){
        (*paramVector->_mat)[iData]=-1.0;
    }

    // Loop over data space
	#pragma omp parallel for
    for (int ixData=_fat; ixData<_nxData-_fat; ixData++){

        double error=100000;
        for (int iv=0; iv<nv; iv++){

            double vValue=ov+iv*dv;
            double xInterp = 0;
            for (int ixModel=0; ixModel<_nxModel; ixModel++){

                double sx=vValue-(*_xKnots->_mat)[ixModel];
                double xWeight=0.0;

                //////////// Compute interpolation weight //////////////////
                // Order 0
                if (_xOrder==0){
                    if (sx>=0.0 && sx<_dkx){
                        xWeight = 1.0;
                    }
                }
                // Order 1
                else if (_xOrder==1){
                    if (ixModel==0) {
                        if (sx>=0 && sx<_dkx){
                            xWeight=1.0-sx/_dkx;
                        }
                    }
                    else if (ixModel>=1 && ixModel < _nxModel-1){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=2.0-sx/_dkx;
                        }
                    }
                    else if (_nxModel-1){
                        if (sx>=0.0 && sx<=_dkx){
                            xWeight=sx/_dkx;
                        }
                    }
                }
                // Order 2
                else if (_xOrder==2){
                    // i = 0
                    if (ixModel==0){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx/(_dkx*_dkx)-2.0*sx/_dkx+1.0;
                        }
                    // i = 1
                    } else if (ixModel==1){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=-3.0/2.0*sx*sx/(_dkx*_dkx)+2.0*sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=sx*sx/(2.0*_dkx*_dkx)-2.0*sx/_dkx+2;
                        }
                    // Main loop
                    } else if (ixModel>=2 && ixModel<_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx/(2.0*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=-sx*sx/(_dkx*_dkx)+3.0*sx/_dkx-1.5;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight=sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+4.5;
                        }
                    // i=nControlPoint-2
                    } else if (ixModel==_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx/(2.0*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0;
                        }
                    // i=nControlPoint-1
                    } else if (ixModel==_nxModel-1){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx/(_dkx*_dkx);
                        }
                    }
                }
                // Order 3
                else if (_xOrder==3){
                    // i = 0
                    if (ixModel==0){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
                        }
                    // i = 1
                    } else if (ixModel==1){
                        if (sx>=0 && sx<_dkx){
                            xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
                        }
                    // i = 2
                    } else if (ixModel==2){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
                        }
                    // Main loop
                    } else if (ixModel>=3 && ixModel<_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight = -sx*sx*sx/(2.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 2.0*sx/_dkx + 2.0/3.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight = 1/(2.0*_dkx*_dkx*_dkx)*sx*sx*sx - 4.0/(_dkx*_dkx)*sx*sx + 10.0*sx/_dkx -22.0/3.0;
                        } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
                            xWeight = -sx*sx*sx/(6.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 8.0*sx/_dkx + 32.0/3.0;
                        }
                    // i=_nxModel-3
                    } else if (ixModel==_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
                        } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
                            xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
                            xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
                        }
                    // i=_nxModel-2
                    } else if (ixModel==_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
                            xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
                            xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
                        }
                    // i=_nxModel-1
                    } else if (ixModel==_nxModel-1){
                        if (sx>=0.0 && sx<=_dkx){
                            xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
                        }
                    }
                }
                ////////////////////////////////////////////////////////////

                // Add contribution of model point
                xInterp+=xWeight*(*_xModel->_mat)[ixModel][0];

            }
            // Finished computing interpolated position for this u-value
            // Update the optimal u-value if interpolated point is clsoer to data point
            if (std::abs(xInterp-(*_xData->_mat)[ixData][0]) < error) {
                error=std::abs(xInterp-(*_xData->_mat)[ixData][0]);
                (*paramVector->_mat)[ixData]=vValue;
            }
        }
        // Finished computing interpolated values for all u's
        if (std::abs(error)>_xTolerance){
            std::cout << "**** ERROR: Could not find a parameter for data point in the x-direction #" << ixData << " " << (*_xData->_mat)[ixData][0]<< " [km]. Try increasing the number of samples! ****" << std::endl;
            std::cout << "Error = " << error << std::endl;
            std::cout << "Tolerance = " << _xTolerance << " [km]" << std::endl;
            assert(1==2);
        }
    }
    return paramVector;
}

// Scaling vector
std::shared_ptr<double2DReg> interpBSpline2dDouble::computeScaleVector(int scalingOn){

    // Variables declaration
    double uValue, vValue, zWeight, xWeight;
    std::shared_ptr<double2DReg> scaleVector, scaleVectorData;
    scaleVector = std::make_shared<double2DReg>(_nzModel, _nxModel);
    scaleVectorData = std::make_shared<double2DReg>(_nzData, _nxData);
    scaleVector->scale(0.0);
    scaleVectorData->scale(0.0);

    for (int ixModel=0; ixModel<_nxModel; ixModel++){
        for (int izModel=0; izModel<_nzModel; izModel++){
            (*scaleVector->_mat)[ixModel][izModel]=1.0;
        }
    }

    // for (int ixData=_fat; ixData<_nxData-_fat; ixData++){
    //     for (int izData=_fat; izData<_nzData-_fat; izData++){
    //         (*scaleVectorData->_mat)[ixData][izData]=1.0;
    //     }
    // }

    if (scalingOn == 1){

        // Apply one forward
        forwardNoScale(false, scaleVector, scaleVectorData);
        // Apply one adjoint
        adjointNoScale(false, scaleVector, scaleVectorData);
        // Compute scaling
        #pragma omp parallel for collapse(2)
        for (int ixModel=0; ixModel<_nxModel; ixModel++){
            for (int izModel=0; izModel<_nzModel; izModel++){
                (*scaleVector->_mat)[ixModel][izModel]=1.0/sqrt((*scaleVector->_mat)[ixModel][izModel]);
            }
        }

        // // Apply one forward
        // adjointNoScale(false, scaleVector, scaleVectorData);
        // // Apply one adjoint
        // forwardNoScale(false, scaleVector, scaleVectorData);
        // // Compute scaling
        // #pragma omp parallel for collapse(2)
        // for (int ixData=_fat; ixData<_nxData-_fat; ixData++){
        //     for (int izData=_fat; izData<_nzData-_fat; izData++){
        //         (*scaleVectorData->_mat)[ixData][izData]=1.0/sqrt((*scaleVectorData->_mat)[ixData][izData]);
        //     }
        // }
    }
    return scaleVector;
    // return scaleVectorData;
}

// Compute (sample) entire surface
std::shared_ptr<double2DReg> interpBSpline2dDouble::computeSurface(int nu, int nv, std::shared_ptr<double2DReg> valModel) {

    int nxControlPoints=_nxModel;
    int nzControlPoints=_nzModel;

    double ou=0.0, ov=0.0;
    double fu=1.0, fv=1.0;
    double du=(fu-ou)/(nu-1);
    double dv=(fv-ov)/(nv-1);
    axis uAxis = axis(nu, ou, du);
    axis vAxis = axis(nv, ov, dv);

    std::cout << "ou = " << ou << std::endl;
    std::cout << "du = " << du << std::endl;
    std::cout << "fu = " << ou+(nu-1)*du << std::endl;

    std::cout << "ov = " << ov << std::endl;
    std::cout << "dv = " << dv << std::endl;
    std::cout << "fv = " << ov+(nv-1)*dv << std::endl;

    _interpSurfaceZ = std::make_shared<double2DReg>(uAxis, vAxis);
    _interpSurfaceX = std::make_shared<double2DReg>(uAxis, vAxis);
    _interpSurfaceVel = std::make_shared<double2DReg>(uAxis, vAxis);
    _interpSurfaceZ->scale(0.0);
    _interpSurfaceX->scale(0.0);
    _interpSurfaceVel->scale(0.0);

	#pragma omp parallel for collapse(2)
    // #pragma omp parallel for
    for (int iv=0; iv<nv; iv++){
        for (int iu=0; iu<nu; iu++){
            double v = ov+dv*iv; // Compute v-parameter value
    	    double u = ou+du*iu; // Compute u-parameter value

            // Start interpolation
            for (int ixModel=0; ixModel<_nxModel; ixModel++){
                // double vWeight=bspline1d(ixModel, v, _xOrder, _dkx, _nxModel, _xKnots);

                ////////////////////////////////////////////////////////////////
                //////////// Compute interpolation weight in x /////////////////
                ////////////////////////////////////////////////////////////////

                double sx=v-(*_xKnots->_mat)[ixModel];
                double xWeight=0.0;

                // Order 0
                if (_xOrder == 0){
                    if (sx>=0.0 && sx<_dkx){
                        xWeight = 1.0;
                    }
                }
                // Order 1
                if (_xOrder == 1){
                    if (ixModel==0) {
                        if (sx>=0 && sx<_dkx){
                            xWeight=1.0-sx/_dkx;
                        }
                    }
                    else if (ixModel>=1 && ixModel < _nxModel-1){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=2.0-sx/_dkx;
                        }
                    }
                    else if (_nxModel-1){
                        if (sx>=0.0 && sx<=_dkx){
                            xWeight=sx/_dkx;
                        }
                    }
                }

                // Order 2
                if (_xOrder == 2){
                    // i = 0
                    if (ixModel==0){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx/(_dkx*_dkx)-2.0*sx/_dkx+1.0;
                        }
                    // i = 1
                    } else if (ixModel==1){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=-3.0/2.0*sx*sx/(_dkx*_dkx)+2.0*sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=sx*sx/(2.0*_dkx*_dkx)-2.0*sx/_dkx+2;
                        }
                    // Main loop
                    } else if (ixModel>=2 && ixModel<_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx/(2.0*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=-sx*sx/(_dkx*_dkx)+3.0*sx/_dkx-1.5;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight=sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+4.5;
                        }
                    // i=nControlPoint-2
                    } else if (ixModel==_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx/(2.0*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0;
                        }
                    // i=nControlPoint-1
                    } else if (ixModel==_nxModel-1){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx/(_dkx*_dkx);
                        }
                    }
                }

                // Order 3
                if (_xOrder == 3){
                    // i = 0
                    if (ixModel==0){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
                        }
                    // i = 1
                    } else if (ixModel==1){
                        if (sx>=0 && sx<_dkx){
            				xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
            			} else if (sx>=_dkx && sx<(2.0*_dkx)){
            				xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
            			}
                    // i = 2
                    } else if (ixModel==2){
                        if (sx>=0 && sx<_dkx){
            				xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
            			} else if (sx>=_dkx && sx<(2.0*_dkx)){
            				xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
            			} else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
            				xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
            			}
                    // Main loop
                    } else if (ixModel>=3 && ixModel<_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight = -sx*sx*sx/(2.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 2.0*sx/_dkx + 2.0/3.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight = 1/(2.0*_dkx*_dkx*_dkx)*sx*sx*sx - 4.0/(_dkx*_dkx)*sx*sx + 10.0*sx/_dkx -22.0/3.0;
                        } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
                            xWeight = -sx*sx*sx/(6.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 8.0*sx/_dkx + 32.0/3.0;
                        }
                    // i=_nxModel-3
                    } else if (ixModel==_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
            				xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
            			} else if(sx>=_dkx && sx<(2.0*_dkx)) {
            				xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
            			} else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
            				xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
            				xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
            			}
                    // i=_nxModel-2
                    } else if (ixModel==_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
            				xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
            			} else if(sx>=_dkx && sx<=(2.0*_dkx)) {
            				xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
            				xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
            			}
                    // i=_nxModel-1
                    } else if (ixModel==_nxModel-1){
                        if (sx>=0.0 && sx<=_dkx){
            				xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
            			}
                    }
                }

                ////////////////////////////////////////////////////////////////

                for (int izModel=0; izModel<_nzModel; izModel++){

                    // double uWeight=bspline1d(izModel, u, _zOrder, _dkz, _nzModel, _zKnots);
                    ////////////////////////////////////////////////////////////////
                    //////////////// Compute interpolation weight in z /////////////
                    ////////////////////////////////////////////////////////////////

                    double sz=u-(*_zKnots->_mat)[izModel];
                    double zWeight=0.0;

                    // Order 0
                    if (_zOrder == 0){
                        if (sz>=0.0 && sz<_dkz){
                            zWeight = 1.0;
                        }
                    }
                    // Order 1
                    if (_zOrder == 1){
                        if (izModel==0) {
                            if (sz>=0 && sz<_dkz){
                                zWeight=1.0-sz/_dkz;
                            }
                        }
                        else if (izModel>=1 && izModel < _nzModel-1){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz/_dkz;
                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                zWeight=2.0-sz/_dkz;
                            }
                        }
                        else if (_nzModel-1){
                            if (sz>=0.0 && sz<=_dkz){
                                zWeight=sz/_dkz;
                            }
                        }
                    }

                    // Order 2
                    if (_zOrder == 2){
                        // i = 0
                        if (izModel==0){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz/(_dkz*_dkz)-2.0*sz/_dkz+1.0;
                            }
                        // i = 1
                        } else if (izModel==1){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=-3.0/2.0*sz*sz/(_dkz*_dkz)+2.0*sz/_dkz;
                            } else if (sz>=_dkz && sz<(2.0*_dkz)) {
                                zWeight=sz*sz/(2.0*_dkz*_dkz)-2.0*sz/_dkz+2;
                            }
                        // Main loop
                        } else if (izModel>=2 && izModel<_nzModel-2){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz/(2.0*_dkz*_dkz);
                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                zWeight=-sz*sz/(_dkz*_dkz)+3.0*sz/_dkz-1.5;
                            } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                zWeight=sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+4.5;
                            }
                        // i=nControlPoint-2
                        } else if (izModel==_nzModel-2){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz/(2.0*_dkz*_dkz);
                            } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                                zWeight=-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0;
                            }
                        // i=nControlPoint-1
                        } else if (izModel==_nzModel-1){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz/(_dkz*_dkz);
                            }
                        }
                    }

                    // Order 3
                    if (_zOrder == 3){
                        // i = 0
                        if (izModel==0){
                            if (sz>=0 && sz<_dkz){
                                zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
                            }
                        // i = 1
                        } else if (izModel==1){
                            if (sz>=0 && sz<_dkz){
                				zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
                			} else if (sz>=_dkz && sz<(2.0*_dkz)){
                				zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
                			}
                        // i = 2
                        } else if (izModel==2){
                            if (sz>=0 && sz<_dkz){
                				zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
                			} else if (sz>=_dkz && sz<(2.0*_dkz)){
                				zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
                			} else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                				zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
                			}
                        // Main loop
                        } else if (izModel>=3 && izModel<_nzModel-3){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                zWeight = -sz*sz*sz/(2.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 2.0*sz/_dkz + 2.0/3.0;
                            } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                zWeight = 1/(2.0*_dkz*_dkz*_dkz)*sz*sz*sz - 4.0/(_dkz*_dkz)*sz*sz + 10.0*sz/_dkz -22.0/3.0;
                            } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
                                zWeight = -sz*sz*sz/(6.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 8.0*sz/_dkz + 32.0/3.0;
                            }
                        // i=_nzModel-3
                        } else if (izModel==_nzModel-3){
                            if (sz>=0.0 && sz<_dkz){
                				zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                			} else if(sz>=_dkz && sz<(2.0*_dkz)) {
                				zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
                			} else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
                				zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
                				zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
                			}
                        // i=_nzModel-2
                        } else if (izModel==_nzModel-2){
                            if (sz>=0.0 && sz<_dkz){
                				zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
                			} else if(sz>=_dkz && sz<=(2.0*_dkz)) {
                				zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
                				zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
                			}
                        // i=_nzModel-1
                        } else if (izModel==_nzModel-1){
                            if (sz>=0.0 && sz<=_dkz){
                				zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
                			}
                        }
                    }
                    ////////////////////////////////////////////////////////////////

                    (*_interpSurfaceX->_mat)[iv][iu] += xWeight*zWeight*(*_xModel->_mat)[ixModel][izModel]; // Interpolate x-position
                    (*_interpSurfaceZ->_mat)[iv][iu] += xWeight*zWeight*(*_zModel->_mat)[ixModel][izModel]; // Interpolate y-position
                    (*_interpSurfaceVel->_mat)[iv][iu] += xWeight*zWeight*(*valModel->_mat)[ixModel][izModel]; // Interpolate velocity
                }
            }
        }
    }
    return _interpSurfaceVel;
}

// Set parameter vectors
void interpBSpline2dDouble::setParamVectors(std::shared_ptr<double1DReg> zParamVector, std::shared_ptr<double1DReg> xParamVector){

    // Set new parameter vectors
    _zParamVector = zParamVector;
    _xParamVector = xParamVector;

    // Recompute scale for amplitude balancing
    _scaleVector = computeScaleVector(_scaling);

}

// Forward
void interpBSpline2dDouble::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const {

    // Forward: Coarse grid to fine grid
    // Model can be on an irregular grid
	if (!add) data->scale(0.0);

    // Loop over data (fine sampling grid)
	#pragma omp parallel for collapse(2)
    for (int ixData=_fat; ixData<_nxData-_fat; ixData++){
        for (int izData=_fat; izData<_nzData-_fat; izData++){

            double uValue = (*_zParamVector->_mat)[izData];
            double vValue = (*_xParamVector->_mat)[ixData];
            for (int ixModel=0; ixModel<_nxModel; ixModel++){

                double sx=vValue-(*_xKnots->_mat)[ixModel];
                double xWeight=0.0;
                //
                // //////////// Compute x-interpolation weight ////////////////////
                // // Order 0
                // if (_xOrder==0){
                //     if (sx>=0.0 && sx<_dkx){
                //         xWeight = 1.0;
                //     }
                // }
                // // Order 1
                // else if (_xOrder==1){
                //     if (ixModel==0) {
                //         if (sx>=0 && sx<_dkx){
                //             xWeight=1.0-sx/_dkx;
                //         }
                //     }
                //     else if (ixModel>=1 && ixModel < _nxModel-1){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx/_dkx;
                //         } else if (sx>=_dkx && sx<(2.0*_dkx)){
                //             xWeight=2.0-sx/_dkx;
                //         }
                //     }
                //     else if (_nxModel-1){
                //         if (sx>=0.0 && sx<=_dkx){
                //             xWeight=sx/_dkx;
                //         }
                //     }
                // }
                // // Order 2
                // else if (_xOrder==2){
                //     // i = 0
                //     if (ixModel==0){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx*sx/(_dkx*_dkx)-2.0*sx/_dkx+1.0;
                //         }
                //     // i = 1
                //     } else if (ixModel==1){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=-3.0/2.0*sx*sx/(_dkx*_dkx)+2.0*sx/_dkx;
                //         } else if (sx>=_dkx && sx<(2.0*_dkx)) {
                //             xWeight=sx*sx/(2.0*_dkx*_dkx)-2.0*sx/_dkx+2;
                //         }
                //     // Main loop
                //     } else if (ixModel>=2 && ixModel<_nxModel-2){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx*sx/(2.0*_dkx*_dkx);
                //         } else if (sx>=_dkx && sx<(2.0*_dkx)){
                //             xWeight=-sx*sx/(_dkx*_dkx)+3.0*sx/_dkx-1.5;
                //         } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                //             xWeight=sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+4.5;
                //         }
                //     // i=nControlPoint-2
                //     } else if (ixModel==_nxModel-2){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx*sx/(2.0*_dkx*_dkx);
                //         } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                //             xWeight=-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0;
                //         }
                //     // i=nControlPoint-1
                //     } else if (ixModel==_nxModel-1){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx*sx/(_dkx*_dkx);
                //         }
                //     }
                // }
                // Order 3
                // else if (_xOrder==3){
                    // i = 0
                    if (ixModel==0){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
                        }
                    // i = 1
                    } else if (ixModel==1){
                        if (sx>=0 && sx<_dkx){
                            xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
                        }
                    // i = 2
                    } else if (ixModel==2){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
                        }
                    // Main loop
                    } else if (ixModel>=3 && ixModel<_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight = -sx*sx*sx/(2.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 2.0*sx/_dkx + 2.0/3.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight = 1/(2.0*_dkx*_dkx*_dkx)*sx*sx*sx - 4.0/(_dkx*_dkx)*sx*sx + 10.0*sx/_dkx -22.0/3.0;
                        } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
                            xWeight = -sx*sx*sx/(6.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 8.0*sx/_dkx + 32.0/3.0;
                        }
                    // i=_nxModel-3
                    } else if (ixModel==_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
                        } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
                            xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
                            xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
                        }
                    // i=_nxModel-2
                    } else if (ixModel==_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
                            xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
                            xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
                        }
                    // i=_nxModel-1
                    } else if (ixModel==_nxModel-1){
                        if (sx>=0.0 && sx<=_dkx){
                            xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
                        }
                    }
                // }
                ////////////////////////////////////////////////////////////////

                for (int izModel=0; izModel<_nzModel; izModel++){
                    double sz=uValue-(*_zKnots->_mat)[izModel];
                    double zWeight=0.0;

                    //////////// Compute z-interpolation weight ////////////////
                    // Order 0
                    // if (_zOrder==0){
                    //     if (sz>=0.0 && sz<_dkz){
                    //         zWeight = 1.0;
                    //     }
                    // }
                    // // Order 1
                    // else if (_zOrder==1){
                    //     if (izModel==0) {
                    //         if (sz>=0 && sz<_dkz){
                    //             zWeight=1.0-sz/_dkz;
                    //         }
                    //     }
                    //     else if (izModel>=1 && izModel < _nzModel-1){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz/_dkz;
                    //         } else if (sz>=_dkz && sz<(2.0*_dkz)){
                    //             zWeight=2.0-sz/_dkz;
                    //         }
                    //     }
                    //     else if (_nzModel-1){
                    //         if (sz>=0.0 && sz<=_dkz){
                    //             zWeight=sz/_dkz;
                    //         }
                    //     }
                    // }
                    // // Order 2
                    // else if (_zOrder==2){
                    //     // i = 0
                    //     if (izModel==0){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz*sz/(_dkz*_dkz)-2.0*sz/_dkz+1.0;
                    //         }
                    //     // i = 1
                    //     } else if (izModel==1){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=-3.0/2.0*sz*sz/(_dkz*_dkz)+2.0*sz/_dkz;
                    //         } else if (sz>=_dkz && sz<(2.0*_dkz)) {
                    //             zWeight=sz*sz/(2.0*_dkz*_dkz)-2.0*sz/_dkz+2;
                    //         }
                    //     // Main loop
                    //     } else if (izModel>=2 && izModel<_nzModel-2){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz*sz/(2.0*_dkz*_dkz);
                    //         } else if (sz>=_dkz && sz<(2.0*_dkz)){
                    //             zWeight=-sz*sz/(_dkz*_dkz)+3.0*sz/_dkz-1.5;
                    //         } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                    //             zWeight=sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+4.5;
                    //         }
                    //     // i=nControlPoint-2
                    //     } else if (izModel==_nzModel-2){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz*sz/(2.0*_dkz*_dkz);
                    //         } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                    //             zWeight=-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0;
                    //         }
                    //     // i=nControlPoint-1
                    //     } else if (izModel==_nzModel-1){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz*sz/(_dkz*_dkz);
                    //         }
                    //     }
                    // }
                    // Order 3
                    // else if (_zOrder==3){
                        // i = 0
                        if (izModel==0){
                            if (sz>=0 && sz<_dkz){
                                zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
                            }
                        // i = 1
                        } else if (izModel==1){
                            if (sz>=0 && sz<_dkz){
                                zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
                            }
                        // i = 2
                        } else if (izModel==2){
                            if (sz>=0 && sz<_dkz){
                                zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
                            } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
                            }
                        // Main loop
                        } else if (izModel>=3 && izModel<_nzModel-3){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                zWeight = -sz*sz*sz/(2.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 2.0*sz/_dkz + 2.0/3.0;
                            } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                zWeight = 1/(2.0*_dkz*_dkz*_dkz)*sz*sz*sz - 4.0/(_dkz*_dkz)*sz*sz + 10.0*sz/_dkz -22.0/3.0;
                            } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
                                zWeight = -sz*sz*sz/(6.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 8.0*sz/_dkz + 32.0/3.0;
                            }
                        // i=_nzModel-3
                        } else if (izModel==_nzModel-3){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                            } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                                zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
                            } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
                                zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
                                zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
                            }
                        // i=_nzModel-2
                        } else if (izModel==_nzModel-2){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
                            } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
                                zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
                                zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
                            }
                        // i=_nzModel-1
                        } else if (izModel==_nzModel-1){
                            if (sz>=0.0 && sz<=_dkz){
                                zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
                            }
                        }
                    // }
                    ////////////////////////////////////////////////////////////

                    // Add contribution to interpolated value (data)
                    (*data->_mat)[ixData][izData] += xWeight*zWeight*(*_scaleVector->_mat)[ixModel][izModel]*(*model->_mat)[ixModel][izModel];
                    // (*data->_mat)[ixData][izData] += xWeight*zWeight*(*_scaleVector->_mat)[ixData][izData]*(*model->_mat)[ixModel][izModel];
                }
            }
        }
    }
}

// Forward no scale
void interpBSpline2dDouble::forwardNoScale(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const {

    // Forward: Coarse grid to fine grid
    // Model can be on an irregular grid
	if (!add) data->scale(0.0);

    // Loop over data (fine sampling grid)
	#pragma omp parallel for collapse(2)
    for (int ixData=_fat; ixData<_nxData-_fat; ixData++){
        for (int izData=_fat; izData<_nzData-_fat; izData++){

            double uValue = (*_zParamVector->_mat)[izData];
            double vValue = (*_xParamVector->_mat)[ixData];
            for (int ixModel=0; ixModel<_nxModel; ixModel++){

                double sx=vValue-(*_xKnots->_mat)[ixModel];
                double xWeight=0.0;

                //////////// Compute x-interpolation weight ////////////////////
                // Order 0
                // if (_xOrder==0){
                //     if (sx>=0.0 && sx<_dkx){
                //         xWeight = 1.0;
                //     }
                // }
                // // Order 1
                // else if (_xOrder==1){
                //     if (ixModel==0) {
                //         if (sx>=0 && sx<_dkx){
                //             xWeight=1.0-sx/_dkx;
                //         }
                //     }
                //     else if (ixModel>=1 && ixModel < _nxModel-1){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx/_dkx;
                //         } else if (sx>=_dkx && sx<(2.0*_dkx)){
                //             xWeight=2.0-sx/_dkx;
                //         }
                //     }
                //     else if (_nxModel-1){
                //         if (sx>=0.0 && sx<=_dkx){
                //             xWeight=sx/_dkx;
                //         }
                //     }
                // }
                // // Order 2
                // else if (_xOrder==2){
                //     // i = 0
                //     if (ixModel==0){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx*sx/(_dkx*_dkx)-2.0*sx/_dkx+1.0;
                //         }
                //     // i = 1
                //     } else if (ixModel==1){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=-3.0/2.0*sx*sx/(_dkx*_dkx)+2.0*sx/_dkx;
                //         } else if (sx>=_dkx && sx<(2.0*_dkx)) {
                //             xWeight=sx*sx/(2.0*_dkx*_dkx)-2.0*sx/_dkx+2;
                //         }
                //     // Main loop
                //     } else if (ixModel>=2 && ixModel<_nxModel-2){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx*sx/(2.0*_dkx*_dkx);
                //         } else if (sx>=_dkx && sx<(2.0*_dkx)){
                //             xWeight=-sx*sx/(_dkx*_dkx)+3.0*sx/_dkx-1.5;
                //         } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                //             xWeight=sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+4.5;
                //         }
                //     // i=nControlPoint-2
                //     } else if (ixModel==_nxModel-2){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx*sx/(2.0*_dkx*_dkx);
                //         } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                //             xWeight=-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0;
                //         }
                //     // i=nControlPoint-1
                //     } else if (ixModel==_nxModel-1){
                //         if (sx>=0.0 && sx<_dkx){
                //             xWeight=sx*sx/(_dkx*_dkx);
                //         }
                //     }
                // }
                // Order 3
                // else if (_xOrder==3){
                    // i = 0
                    if (ixModel==0){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
                        }
                    // i = 1
                    } else if (ixModel==1){
                        if (sx>=0 && sx<_dkx){
                            xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
                        }
                    // i = 2
                    } else if (ixModel==2){
                        if (sx>=0 && sx<_dkx){
                            xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
                        }
                    // Main loop
                    } else if (ixModel>=3 && ixModel<_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if (sx>=_dkx && sx<(2.0*_dkx)){
                            xWeight = -sx*sx*sx/(2.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 2.0*sx/_dkx + 2.0/3.0;
                        } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                            xWeight = 1/(2.0*_dkx*_dkx*_dkx)*sx*sx*sx - 4.0/(_dkx*_dkx)*sx*sx + 10.0*sx/_dkx -22.0/3.0;
                        } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
                            xWeight = -sx*sx*sx/(6.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 8.0*sx/_dkx + 32.0/3.0;
                        }
                    // i=_nxModel-3
                    } else if (ixModel==_nxModel-3){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                            xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
                        } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
                            xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
                            xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
                        }
                    // i=_nxModel-2
                    } else if (ixModel==_nxModel-2){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
                        } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
                            xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
                            xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
                        }
                    // i=_nxModel-1
                    } else if (ixModel==_nxModel-1){
                        if (sx>=0.0 && sx<=_dkx){
                            xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
                        }
                    }
                // }
                ////////////////////////////////////////////////////////////////

                for (int izModel=0; izModel<_nzModel; izModel++){
                    double sz=uValue-(*_zKnots->_mat)[izModel];
                    double zWeight=0.0;

                    //////////// Compute z-interpolation weight ////////////////
                    // Order 0
                    // if (_zOrder==0){
                    //     if (sz>=0.0 && sz<_dkz){
                    //         zWeight = 1.0;
                    //     }
                    // }
                    // // Order 1
                    // else if (_zOrder==1){
                    //     if (izModel==0) {
                    //         if (sz>=0 && sz<_dkz){
                    //             zWeight=1.0-sz/_dkz;
                    //         }
                    //     }
                    //     else if (izModel>=1 && izModel < _nzModel-1){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz/_dkz;
                    //         } else if (sz>=_dkz && sz<(2.0*_dkz)){
                    //             zWeight=2.0-sz/_dkz;
                    //         }
                    //     }
                    //     else if (_nzModel-1){
                    //         if (sz>=0.0 && sz<=_dkz){
                    //             zWeight=sz/_dkz;
                    //         }
                    //     }
                    // }
                    // // Order 2
                    // else if (_zOrder==2){
                    //     // i = 0
                    //     if (izModel==0){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz*sz/(_dkz*_dkz)-2.0*sz/_dkz+1.0;
                    //         }
                    //     // i = 1
                    //     } else if (izModel==1){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=-3.0/2.0*sz*sz/(_dkz*_dkz)+2.0*sz/_dkz;
                    //         } else if (sz>=_dkz && sz<(2.0*_dkz)) {
                    //             zWeight=sz*sz/(2.0*_dkz*_dkz)-2.0*sz/_dkz+2;
                    //         }
                    //     // Main loop
                    //     } else if (izModel>=2 && izModel<_nzModel-2){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz*sz/(2.0*_dkz*_dkz);
                    //         } else if (sz>=_dkz && sz<(2.0*_dkz)){
                    //             zWeight=-sz*sz/(_dkz*_dkz)+3.0*sz/_dkz-1.5;
                    //         } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                    //             zWeight=sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+4.5;
                    //         }
                    //     // i=nControlPoint-2
                    //     } else if (izModel==_nzModel-2){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz*sz/(2.0*_dkz*_dkz);
                    //         } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                    //             zWeight=-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0;
                    //         }
                    //     // i=nControlPoint-1
                    //     } else if (izModel==_nzModel-1){
                    //         if (sz>=0.0 && sz<_dkz){
                    //             zWeight=sz*sz/(_dkz*_dkz);
                    //         }
                    //     }
                    // }
                    // Order 3
                    // else if (_zOrder==3){
                        // i = 0
                        if (izModel==0){
                            if (sz>=0 && sz<_dkz){
                                zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
                            }
                        // i = 1
                        } else if (izModel==1){
                            if (sz>=0 && sz<_dkz){
                                zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
                            }
                        // i = 2
                        } else if (izModel==2){
                            if (sz>=0 && sz<_dkz){
                                zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
                            } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
                            }
                        // Main loop
                        } else if (izModel>=3 && izModel<_nzModel-3){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                            } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                zWeight = -sz*sz*sz/(2.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 2.0*sz/_dkz + 2.0/3.0;
                            } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                zWeight = 1/(2.0*_dkz*_dkz*_dkz)*sz*sz*sz - 4.0/(_dkz*_dkz)*sz*sz + 10.0*sz/_dkz -22.0/3.0;
                            } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
                                zWeight = -sz*sz*sz/(6.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 8.0*sz/_dkz + 32.0/3.0;
                            }
                        // i=_nzModel-3
                        } else if (izModel==_nzModel-3){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                            } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                                zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
                            } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
                                zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
                                zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
                            }
                        // i=_nzModel-2
                        } else if (izModel==_nzModel-2){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
                            } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
                                zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
                                zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
                            }
                        // i=_nzModel-1
                        } else if (izModel==_nzModel-1){
                            if (sz>=0.0 && sz<=_dkz){
                                zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
                            }
                        }
                    // }
                    ////////////////////////////////////////////////////////////

                    // Add contribution to interpolated value (data)
                    (*data->_mat)[ixData][izData] += xWeight*zWeight*(*model->_mat)[ixModel][izModel];

                }
            }
        }
    }
}

// Adjoint
void interpBSpline2dDouble::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const {

    // Adjoint: Fine grid to coarse grid
    // Model can be on an irregular grid
    if (!add) model->scale(0.0);

        // Loop over data (fine sampling grid)
        #pragma omp parallel for collapse(2)
        for (int ixModel=0; ixModel<_nxModel; ixModel++){
            for (int izModel=0; izModel<_nzModel; izModel++){

                for (int ixData=_fat; ixData<_nxData-_fat; ixData++){
                    double vValue = (*_xParamVector->_mat)[ixData];
                    double sx=vValue-(*_xKnots->_mat)[ixModel];
                    double xWeight=0.0;

                    //////////// Compute x-interpolation weight ////////////////////
                    // Order 0
                    if (_xOrder==0){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight = 1.0;
                        }
                    }
                    // Order 1
                    else if (_xOrder==1){
                        if (ixModel==0) {
                            if (sx>=0 && sx<_dkx){
                                xWeight=1.0-sx/_dkx;
                            }
                        }
                        else if (ixModel>=1 && ixModel < _nxModel-1){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx/_dkx;
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight=2.0-sx/_dkx;
                            }
                        }
                        else if (_nxModel-1){
                            if (sx>=0.0 && sx<=_dkx){
                                xWeight=sx/_dkx;
                            }
                        }
                    }
                    // Order 2
                    else if (_xOrder==2){
                        // i = 0
                        if (ixModel==0){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx/(_dkx*_dkx)-2.0*sx/_dkx+1.0;
                            }
                        // i = 1
                        } else if (ixModel==1){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=-3.0/2.0*sx*sx/(_dkx*_dkx)+2.0*sx/_dkx;
                            } else if (sx>=_dkx && sx<(2.0*_dkx)) {
                                xWeight=sx*sx/(2.0*_dkx*_dkx)-2.0*sx/_dkx+2;
                            }
                        // Main loop
                        } else if (ixModel>=2 && ixModel<_nxModel-2){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx/(2.0*_dkx*_dkx);
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight=-sx*sx/(_dkx*_dkx)+3.0*sx/_dkx-1.5;
                            } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                                xWeight=sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+4.5;
                            }
                        // i=nControlPoint-2
                        } else if (ixModel==_nxModel-2){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx/(2.0*_dkx*_dkx);
                            } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                                xWeight=-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0;
                            }
                        // i=nControlPoint-1
                        } else if (ixModel==_nxModel-1){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx/(_dkx*_dkx);
                            }
                        }
                    }
                    // Order 3
                    else if (_xOrder==3){
                        // i = 0
                        if (ixModel==0){
                            if (sx>=0 && sx<_dkx){
                                xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
                            }
                        // i = 1
                        } else if (ixModel==1){
                            if (sx>=0 && sx<_dkx){
                                xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
                            }
                        // i = 2
                        } else if (ixModel==2){
                            if (sx>=0 && sx<_dkx){
                                xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
                            } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                                xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
                            }
                        // Main loop
                        } else if (ixModel>=3 && ixModel<_nxModel-3){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight = -sx*sx*sx/(2.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 2.0*sx/_dkx + 2.0/3.0;
                            } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                                xWeight = 1/(2.0*_dkx*_dkx*_dkx)*sx*sx*sx - 4.0/(_dkx*_dkx)*sx*sx + 10.0*sx/_dkx -22.0/3.0;
                            } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
                                xWeight = -sx*sx*sx/(6.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 8.0*sx/_dkx + 32.0/3.0;
                            }
                        // i=_nxModel-3
                        } else if (ixModel==_nxModel-3){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                            } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                                xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
                            } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
                                xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
                                xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
                            }
                        // i=_nxModel-2
                        } else if (ixModel==_nxModel-2){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
                            } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
                                xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
                                xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
                            }
                        // i=_nxModel-1
                        } else if (ixModel==_nxModel-1){
                            if (sx>=0.0 && sx<=_dkx){
                                xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
                            }
                        }
                    }
                    ////////////////////////////////////////////////////////////////

                    for (int izData=_fat; izData<_nzData-_fat; izData++){
                        double uValue = (*_zParamVector->_mat)[izData];
                        double sz=uValue-(*_zKnots->_mat)[izModel];
                        double zWeight=0.0;

                        //////////// Compute z-interpolation weight ////////////////
                        // Order 0
                        if (_zOrder==0){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight = 1.0;
                            }
                        }
                        // Order 1
                        else if (_zOrder==1){
                            if (izModel==0) {
                                if (sz>=0 && sz<_dkz){
                                    zWeight=1.0-sz/_dkz;
                                }
                            }
                            else if (izModel>=1 && izModel < _nzModel-1){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz/_dkz;
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=2.0-sz/_dkz;
                                }
                            }
                            else if (_nzModel-1){
                                if (sz>=0.0 && sz<=_dkz){
                                    zWeight=sz/_dkz;
                                }
                            }
                        }
                        // Order 2
                        else if (_zOrder==2){
                            // i = 0
                            if (izModel==0){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz/(_dkz*_dkz)-2.0*sz/_dkz+1.0;
                                }
                            // i = 1
                            } else if (izModel==1){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=-3.0/2.0*sz*sz/(_dkz*_dkz)+2.0*sz/_dkz;
                                } else if (sz>=_dkz && sz<(2.0*_dkz)) {
                                    zWeight=sz*sz/(2.0*_dkz*_dkz)-2.0*sz/_dkz+2;
                                }
                            // Main loop
                            } else if (izModel>=2 && izModel<_nzModel-2){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz/(2.0*_dkz*_dkz);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=-sz*sz/(_dkz*_dkz)+3.0*sz/_dkz-1.5;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    zWeight=sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+4.5;
                                }
                            // i=nControlPoint-2
                            } else if (izModel==_nzModel-2){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz/(2.0*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                                    zWeight=-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0;
                                }
                            // i=nControlPoint-1
                            } else if (izModel==_nzModel-1){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz/(_dkz*_dkz);
                                }
                            }
                        }
                        // Order 3
                        else if (_zOrder==3){
                            // i = 0
                            if (izModel==0){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
                                }
                            // i = 1
                            } else if (izModel==1){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
                                }
                            // i = 2
                            } else if (izModel==2){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
                                }
                            // Main loop
                            } else if (izModel>=3 && izModel<_nzModel-3){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight = -sz*sz*sz/(2.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 2.0*sz/_dkz + 2.0/3.0;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    zWeight = 1/(2.0*_dkz*_dkz*_dkz)*sz*sz*sz - 4.0/(_dkz*_dkz)*sz*sz + 10.0*sz/_dkz -22.0/3.0;
                                } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
                                    zWeight = -sz*sz*sz/(6.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 8.0*sz/_dkz + 32.0/3.0;
                                }
                            // i=_nzModel-3
                            } else if (izModel==_nzModel-3){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                                    zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
                                } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
                                    zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
                                    zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
                                }
                            // i=_nzModel-2
                            } else if (izModel==_nzModel-2){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
                                    zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
                                    zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
                                }
                            // i=_nzModel-1
                            } else if (izModel==_nzModel-1){
                                if (sz>=0.0 && sz<=_dkz){
                                    zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
                                }
                            }
                        }
                        ////////////////////////////////////////////////////////////

                        // Add contribution to interpolated value (data)
                        (*model->_mat)[ixModel][izModel] += xWeight*zWeight*(*_scaleVector->_mat)[ixModel][izModel]*(*data->_mat)[ixData][izData];
                        // (*model->_mat)[ixModel][izModel] += xWeight*zWeight*(*_scaleVector->_mat)[ixData][izData]*(*data->_mat)[ixData][izData];
                    }
                }
            }
        }
    }

// Adjoint no scale
void interpBSpline2dDouble::adjointNoScale(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const {

    // Adjoint: Fine grid to coarse grid
    // Model can be on an irregular grid
	if (!add) model->scale(0.0);

        // Loop over data (fine sampling grid)
    	#pragma omp parallel for collapse(2)
        for (int ixModel=0; ixModel<_nxModel; ixModel++){
            for (int izModel=0; izModel<_nzModel; izModel++){

                for (int ixData=_fat; ixData<_nxData-_fat; ixData++){
                    double vValue = (*_xParamVector->_mat)[ixData];
                    double sx=vValue-(*_xKnots->_mat)[ixModel];
                    double xWeight=0.0;

                    //////////// Compute x-interpolation weight ////////////////////
                    // Order 0
                    if (_xOrder==0){
                        if (sx>=0.0 && sx<_dkx){
                            xWeight = 1.0;
                        }
                    }
                    // Order 1
                    else if (_xOrder==1){
                        if (ixModel==0) {
                            if (sx>=0 && sx<_dkx){
                                xWeight=1.0-sx/_dkx;
                            }
                        }
                        else if (ixModel>=1 && ixModel < _nxModel-1){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx/_dkx;
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight=2.0-sx/_dkx;
                            }
                        }
                        else if (_nxModel-1){
                            if (sx>=0.0 && sx<=_dkx){
                                xWeight=sx/_dkx;
                            }
                        }
                    }
                    // Order 2
                    else if (_xOrder==2){
                        // i = 0
                        if (ixModel==0){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx/(_dkx*_dkx)-2.0*sx/_dkx+1.0;
                            }
                        // i = 1
                        } else if (ixModel==1){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=-3.0/2.0*sx*sx/(_dkx*_dkx)+2.0*sx/_dkx;
                            } else if (sx>=_dkx && sx<(2.0*_dkx)) {
                                xWeight=sx*sx/(2.0*_dkx*_dkx)-2.0*sx/_dkx+2;
                            }
                        // Main loop
                        } else if (ixModel>=2 && ixModel<_nxModel-2){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx/(2.0*_dkx*_dkx);
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight=-sx*sx/(_dkx*_dkx)+3.0*sx/_dkx-1.5;
                            } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                                xWeight=sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+4.5;
                            }
                        // i=nControlPoint-2
                        } else if (ixModel==_nxModel-2){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx/(2.0*_dkx*_dkx);
                            } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                                xWeight=-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0;
                            }
                        // i=nControlPoint-1
                        } else if (ixModel==_nxModel-1){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx/(_dkx*_dkx);
                            }
                        }
                    }
                    // Order 3
                    else if (_xOrder==3){
                        // i = 0
                        if (ixModel==0){
                            if (sx>=0 && sx<_dkx){
                                xWeight=-sx*sx*sx/(_dkx*_dkx*_dkx)+3.0*sx*sx/(_dkx*_dkx)-3.0*sx/_dkx+1.0;
                            }
                        // i = 1
                        } else if (ixModel==1){
                            if (sx>=0 && sx<_dkx){
                                xWeight=7.0*sx*sx*sx/(4.0*_dkx*_dkx*_dkx)-9.0*sx*sx/(2.0*_dkx*_dkx)+3.0*sx/_dkx;
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight=-sx*sx*sx/(4.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-3.0*sx/_dkx+2.0;
                            }
                        // i = 2
                        } else if (ixModel==2){
                            if (sx>=0 && sx<_dkx){
                                xWeight=-11.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx);
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight=7.0*sx*sx*sx/(12.0*_dkx*_dkx*_dkx)-3*sx*sx/(_dkx*_dkx)+9.0*sx/(2.0*_dkx)-3.0/2.0;
                            } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                                xWeight=-sx*sx*sx/(6.0*_dkx*_dkx*_dkx)+3.0*sx*sx/(2.0*_dkx*_dkx)-9.0*sx/(2.0*_dkx)+9.0/2.0;
                            }
                        // Main loop
                        } else if (ixModel>=3 && ixModel<_nxModel-3){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                            } else if (sx>=_dkx && sx<(2.0*_dkx)){
                                xWeight = -sx*sx*sx/(2.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 2.0*sx/_dkx + 2.0/3.0;
                            } else if (sx>=(2.0*_dkx) && sx<(3.0*_dkx)){
                                xWeight = 1/(2.0*_dkx*_dkx*_dkx)*sx*sx*sx - 4.0/(_dkx*_dkx)*sx*sx + 10.0*sx/_dkx -22.0/3.0;
                            } else if (sx>=(3.0*_dkx) && sx<(4.0*_dkx)){
                                xWeight = -sx*sx*sx/(6.0*_dkx*_dkx*_dkx) + 2.0*sx*sx/(_dkx*_dkx) - 8.0*sx/_dkx + 32.0/3.0;
                            }
                        // i=_nxModel-3
                        } else if (ixModel==_nxModel-3){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx*sx/(6.0*_dkx*_dkx*_dkx);
                            } else if(sx>=_dkx && sx<(2.0*_dkx)) {
                                xWeight=-sx*sx*sx/(3.0*_dkx*_dkx*_dkx)+sx*sx/(_dkx*_dkx)-sx/(2*_dkx)+(3.0/2.0-sx/(2.0*_dkx))*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx);
                            } else if(sx>=(2.0*_dkx) && sx<=(3.0*_dkx)) {
                                xWeight=sx/(3.0*_dkx)*(sx*sx/(2.0*_dkx*_dkx)-3*sx/_dkx+9.0/2.0);
                                xWeight+=(3.0/2.0-sx/(2.0*_dkx))*(-3*(sx-_dkx)*(sx-_dkx)/(2.0*_dkx*_dkx)+4*(sx-_dkx)/_dkx-2.0);
                            }
                        // i=_nxModel-2
                        } else if (ixModel==_nxModel-2){
                            if (sx>=0.0 && sx<_dkx){
                                xWeight=sx*sx*sx/(4.0*_dkx*_dkx*_dkx);
                            } else if(sx>=_dkx && sx<=(2.0*_dkx)) {
                                xWeight=sx/(2.0*_dkx)*(-3.0*sx*sx/(2.0*_dkx*_dkx)+4.0*sx/_dkx-2.0);
                                xWeight+=(2.0-sx/_dkx)*(sx-_dkx)*(sx-_dkx)/(_dkx*_dkx);
                            }
                        // i=_nxModel-1
                        } else if (ixModel==_nxModel-1){
                            if (sx>=0.0 && sx<=_dkx){
                                xWeight=sx*sx*sx/(_dkx*_dkx*_dkx);
                            }
                        }
                    }
                    ////////////////////////////////////////////////////////////////

                    for (int izData=_fat; izData<_nzData-_fat; izData++){
                        double uValue = (*_zParamVector->_mat)[izData];
                        double sz=uValue-(*_zKnots->_mat)[izModel];
                        double zWeight=0.0;

                        //////////// Compute z-interpolation weight ////////////////
                        // Order 0
                        if (_zOrder==0){
                            if (sz>=0.0 && sz<_dkz){
                                zWeight = 1.0;
                            }
                        }
                        // Order 1
                        else if (_zOrder==1){
                            if (izModel==0) {
                                if (sz>=0 && sz<_dkz){
                                    zWeight=1.0-sz/_dkz;
                                }
                            }
                            else if (izModel>=1 && izModel < _nzModel-1){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz/_dkz;
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=2.0-sz/_dkz;
                                }
                            }
                            else if (_nzModel-1){
                                if (sz>=0.0 && sz<=_dkz){
                                    zWeight=sz/_dkz;
                                }
                            }
                        }
                        // Order 2
                        else if (_zOrder==2){
                            // i = 0
                            if (izModel==0){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz/(_dkz*_dkz)-2.0*sz/_dkz+1.0;
                                }
                            // i = 1
                            } else if (izModel==1){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=-3.0/2.0*sz*sz/(_dkz*_dkz)+2.0*sz/_dkz;
                                } else if (sz>=_dkz && sz<(2.0*_dkz)) {
                                    zWeight=sz*sz/(2.0*_dkz*_dkz)-2.0*sz/_dkz+2;
                                }
                            // Main loop
                            } else if (izModel>=2 && izModel<_nzModel-2){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz/(2.0*_dkz*_dkz);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=-sz*sz/(_dkz*_dkz)+3.0*sz/_dkz-1.5;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    zWeight=sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+4.5;
                                }
                            // i=nControlPoint-2
                            } else if (izModel==_nzModel-2){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz/(2.0*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                                    zWeight=-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0;
                                }
                            // i=nControlPoint-1
                            } else if (izModel==_nzModel-1){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz/(_dkz*_dkz);
                                }
                            }
                        }
                        // Order 3
                        else if (_zOrder==3){
                            // i = 0
                            if (izModel==0){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=-sz*sz*sz/(_dkz*_dkz*_dkz)+3.0*sz*sz/(_dkz*_dkz)-3.0*sz/_dkz+1.0;
                                }
                            // i = 1
                            } else if (izModel==1){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=7.0*sz*sz*sz/(4.0*_dkz*_dkz*_dkz)-9.0*sz*sz/(2.0*_dkz*_dkz)+3.0*sz/_dkz;
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=-sz*sz*sz/(4.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-3.0*sz/_dkz+2.0;
                                }
                            // i = 2
                            } else if (izModel==2){
                                if (sz>=0 && sz<_dkz){
                                    zWeight=-11.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight=7.0*sz*sz*sz/(12.0*_dkz*_dkz*_dkz)-3*sz*sz/(_dkz*_dkz)+9.0*sz/(2.0*_dkz)-3.0/2.0;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    zWeight=-sz*sz*sz/(6.0*_dkz*_dkz*_dkz)+3.0*sz*sz/(2.0*_dkz*_dkz)-9.0*sz/(2.0*_dkz)+9.0/2.0;
                                }
                            // Main loop
                            } else if (izModel>=3 && izModel<_nzModel-3){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                                } else if (sz>=_dkz && sz<(2.0*_dkz)){
                                    zWeight = -sz*sz*sz/(2.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 2.0*sz/_dkz + 2.0/3.0;
                                } else if (sz>=(2.0*_dkz) && sz<(3.0*_dkz)){
                                    zWeight = 1/(2.0*_dkz*_dkz*_dkz)*sz*sz*sz - 4.0/(_dkz*_dkz)*sz*sz + 10.0*sz/_dkz -22.0/3.0;
                                } else if (sz>=(3.0*_dkz) && sz<(4.0*_dkz)){
                                    zWeight = -sz*sz*sz/(6.0*_dkz*_dkz*_dkz) + 2.0*sz*sz/(_dkz*_dkz) - 8.0*sz/_dkz + 32.0/3.0;
                                }
                            // i=_nzModel-3
                            } else if (izModel==_nzModel-3){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(6.0*_dkz*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<(2.0*_dkz)) {
                                    zWeight=-sz*sz*sz/(3.0*_dkz*_dkz*_dkz)+sz*sz/(_dkz*_dkz)-sz/(2*_dkz)+(3.0/2.0-sz/(2.0*_dkz))*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz);
                                } else if(sz>=(2.0*_dkz) && sz<=(3.0*_dkz)) {
                                    zWeight=sz/(3.0*_dkz)*(sz*sz/(2.0*_dkz*_dkz)-3*sz/_dkz+9.0/2.0);
                                    zWeight+=(3.0/2.0-sz/(2.0*_dkz))*(-3*(sz-_dkz)*(sz-_dkz)/(2.0*_dkz*_dkz)+4*(sz-_dkz)/_dkz-2.0);
                                }
                            // i=_nzModel-2
                            } else if (izModel==_nzModel-2){
                                if (sz>=0.0 && sz<_dkz){
                                    zWeight=sz*sz*sz/(4.0*_dkz*_dkz*_dkz);
                                } else if(sz>=_dkz && sz<=(2.0*_dkz)) {
                                    zWeight=sz/(2.0*_dkz)*(-3.0*sz*sz/(2.0*_dkz*_dkz)+4.0*sz/_dkz-2.0);
                                    zWeight+=(2.0-sz/_dkz)*(sz-_dkz)*(sz-_dkz)/(_dkz*_dkz);
                                }
                            // i=_nzModel-1
                            } else if (izModel==_nzModel-1){
                                if (sz>=0.0 && sz<=_dkz){
                                    zWeight=sz*sz*sz/(_dkz*_dkz*_dkz);
                                }
                            }
                        }
                        ////////////////////////////////////////////////////////////

                        // Add contribution to interpolated value (data)
                        (*model->_mat)[ixModel][izModel] += xWeight*zWeight*(*data->_mat)[ixData][izData];
                    }
                }
            }
        }

}

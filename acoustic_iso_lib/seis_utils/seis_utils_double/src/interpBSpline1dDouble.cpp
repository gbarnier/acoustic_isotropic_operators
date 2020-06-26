#include <double1DReg.h>
#include <iostream>
#include "interpBSpline1dDouble.h"
#include <omp.h>
#include <vector>

// Contructor if you provide the control point positions in a double1DReg
interpBSpline1dDouble::interpBSpline1dDouble(int order, std::shared_ptr<double1DReg> xModel, axis xDataAxis, int nParam, int scaling, double tolerance, int fat){

    // B-spline parameters
    _order = order; // Order of interpolation
    _scaling = scaling; // 1: Compute and apply scaling to balance operator amplitudes

    // Model
    _xModel = xModel; // x-coordinates of control points
    _nModel = _xModel->getHyper()->getAxis(1).n;

    // Data
    _fat = fat;
    _xDataAxis = xDataAxis; // Uniformly distributed data
    _xData = std::make_shared<double1DReg>(_xDataAxis);
    _nData =  _xData->getHyper()->getAxis(1).n;
    for (int iData=0; iData <_xDataAxis.n; iData++){
        (*_xData->_mat)[iData] = _xDataAxis.o+iData*_xDataAxis.d;
    }

    // Set the tolerance [km]
    _tolerance=tolerance*_xDataAxis.d;
    _nParam=nParam;

    // Build the knot vector
    _knots = buildKnotVector();

    // Compute parameter vector
    _paramVector=computeParamVector();

    // Compute scaling vector for amplitude balancing in FWD/ADJ
    _scaleVector = computeScaleVector(scaling);

}

// Function that generates a knot vector between [0,1] (with clamping if order > 0)
std::shared_ptr<double1DReg> interpBSpline1dDouble::buildKnotVector() {

    _nk=_nModel+_order+1; // number of knots
    _nkSimple=_nk-2*_order;
    _ok = 0.0; // Position of FIRST knot
    _fk = 1.0; // Position of LAST knot
    _dk=(_fk-_ok)/(_nkSimple-1); // Knot sampling
    _kAxis = axis(_nk, _ok, _dk); // Knot axis
    std::shared_ptr<double1DReg> knots(new double1DReg(_kAxis)); // Allocate knot vector

    // Compute starting knots with multiplicity > 1 (if order>0)
    for (int ik=0; ik<_order+1; ik++){
		(*knots->_mat)[ik] = _ok;
	}
    // Compute knots with multiplicity 1
	for (int ik=_order+1; ik<_nk-_order; ik++){
        (*knots->_mat)[ik] = _ok+(ik-_order)*_dk;
    }
    // Compute end knots with multiplicity > 1 (if order>0)
	for (int ik=_nk-_order; ik<_nk; ik++){
        (*knots->_mat)[ik] = _fk;
    }
    return knots;
}

// Function that computes the parameter corresponding to each control point positions
std::shared_ptr<double1DReg> interpBSpline1dDouble::computeParamVector() {

    // Generate a vector u in the parameter space with dense sampling
    int nu = _nParam;
    double ou=_ok;
	double fu=_fk;
	double du=(fu-ou)/(nu-1);
	nu=nu-1;
    axis uAxis= axis(nu, ou, du);
    std::shared_ptr<double1DReg> u(new double1DReg(nu));
    std::shared_ptr<double1DReg> paramVector(new double1DReg(_nData));

    // Initialize param vector with -1
    #pragma omp parallel for
    for (int iData=0; iData<_nData; iData++){
        (*paramVector->_mat)[iData]=-1.0;
    }

    // Loop over data space
	#pragma omp parallel for
    for (int iData=_fat; iData<_nData-_fat; iData++){

        double error=100000; // Initialize error with big number

        // Loop over parameter vector
        for (int iu=0; iu<nu; iu++){

            // Compute parameter value
            double uValue=ou+iu*du;
            double xInterp = 0;

            // Compute interpolated position for this parameter value
            for (int iModel=0; iModel<_nModel; iModel++){ // Loop over control points

                ///////////////// Compute B-spline weights /////////////////////
                int iControl=iModel;
                int nControlPoints=_nModel;
                double s=uValue-(*_knots->_mat)[iControl];
                double weight=0;
                double dk=_dk;

                // Order 0
                if (_order == 0){
                    if (s>=0.0 && s<dk){
                        weight = 1.0;
                    }
                }
                // Order 1
                else if (_order == 1){

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
                // Order 2
                else if (_order == 2){

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
                // Order 3
                else if (_order == 3){

                    if (iControl==0){
                        if (s>=0 && s<dk){
                            weight=-s*s*s/(dk*dk*dk)+3.0*s*s/(dk*dk)-3.0*s/dk+1.0;
                        }
                    } else if (iControl==1){
                        if (s>=0 && s<dk){
                            weight=7.0*s*s*s/(4.0*dk*dk*dk)-9.0*s*s/(2.0*dk*dk)+3.0*s/dk;
                        } else if (s>=dk && s<(2.0*dk)){
                            weight=-s*s*s/(4.0*dk*dk*dk)+3.0*s*s/(2.0*dk*dk)-3.0*s/dk+2.0;
                        }
                    } else if (iControl==2){
                        if (s>=0 && s<dk){
                        weight=-11.0*s*s*s/(12.0*dk*dk*dk)+3.0*s*s/(2.0*dk*dk);
                    } else if (s>=dk && s<(2.0*dk)){
                        weight=7.0*s*s*s/(12.0*dk*dk*dk)-3*s*s/(dk*dk)+9.0*s/(2.0*dk)-3.0/2.0;
                    } else if (s>=(2.0*dk) && s<(3.0*dk)){
                        weight=-s*s*s/(6.0*dk*dk*dk)+3.0*s*s/(2.0*dk*dk)-9.0*s/(2.0*dk)+9.0/2.0;
                    }
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
                    } else if (iControl==nControlPoints-3){
                        if (s>=0.0 && s<dk){
                            weight=s*s*s/(6.0*dk*dk*dk);
                        } else if(s>=dk && s<(2.0*dk)) {
                            weight=-s*s*s/(3.0*dk*dk*dk)+s*s/(dk*dk)-s/(2*dk)+(3.0/2.0-s/(2.0*dk))*(s-dk)*(s-dk)/(2.0*dk*dk);
                        } else if(s>=(2.0*dk) && s<(3.0*dk)) {
                            weight=s/(3.0*dk)*(s*s/(2.0*dk*dk)-3*s/dk+9.0/2.0);
                            weight+=(3.0/2.0-s/(2.0*dk))*(-3*(s-dk)*(s-dk)/(2.0*dk*dk)+4*(s-dk)/dk-2.0);
                        }
                    } else if (iControl==nControlPoints-2){
                        if (s>=0.0 && s<dk){
                            weight=s*s*s/(4.0*dk*dk*dk);
                        } else if(s>=dk && s<(2.0*dk)) {
                            weight=s/(2.0*dk)*(-3.0*s*s/(2.0*dk*dk)+4.0*s/dk-2.0);
                            weight+=(2.0-s/dk)*(s-dk)*(s-dk)/(dk*dk);
                        }
                    } else if (iControl==nControlPoints-1){
                        if (s>=0.0 && s<dk){
                            weight=s*s*s/(dk*dk*dk);
                        }
                    }
                }

                // Add contribution of control point
                xInterp+=weight*(*_xModel->_mat)[iModel];
            }
            ////////////////////////////////////////////////////////////////////

            // Compute difference between interpolated position and true position of xData for this u-value
            if (std::abs(xInterp-(*_xData->_mat)[iData]) < error) {
                error=std::abs(xInterp-(*_xData->_mat)[iData]);
                (*paramVector->_mat)[iData]=uValue;
            }
        }

        // Finished computing the values for all parameter vector
        // The optimal parameter for this data point is uValue
        // The difference between the point on the curve for uValue and the data position is stored in error
        // Check if the value for the optimal parameter (paramVector) gives a good enough position
        if (std::abs(error)>_tolerance){
            std::cout << "**** ERROR: Could not find a parameter for data point #" << iData << " " << (*_xData->_mat)[iData]<< " [km]. Try increasing the number of samples! ****" << std::endl;
            std::cout << "Error = " << error <<" [km]" << std::endl;
            std::cout << "Tolerance = " << _tolerance << " [km]" << std::endl;
            throw std::runtime_error("");
        }
    }

    // for (int iData=0; iData<_nData; iData++){
    //     std::cout << "i= " << iData << std::endl;
    //     std::cout << "paramVector= " << (*paramVector->_mat)[iData] << std::endl;
    // }

    return paramVector;
}

// Function that estimates the scaling factors to balance the operator amplitudes
std::shared_ptr<double1DReg> interpBSpline1dDouble::computeScaleVector(int scaling) {

    // Declaration and allocation
    double uValue, weight;
    std::shared_ptr<double1DReg> scaleVector = _xModel->clone();
    std::shared_ptr<double1DReg> scaleVectorData(new double1DReg(_xDataAxis));
    scaleVectorData->scale(0.0);
    for (int iModel=0; iModel<_nModel; iModel++){ // Replace that by ->set()
        (*scaleVector->_mat)[iModel]=1.0;
    }

    if (scaling == 1){

        // Apply forward
        for (int iData=_fat; iData<_nData-_fat; iData++){
            uValue=(*_paramVector->_mat)[iData];
            for (int iModel=0; iModel<_nModel; iModel++){
                weight=bspline1d(iModel, uValue, _order, _dk, _nModel, _knots);
                (*scaleVectorData->_mat)[iData]+=weight*(*scaleVector->_mat)[iModel];
            }
        }
        // Apply adjoint
        scaleVector->scale(0.0);
        for (int iData=_fat; iData<_nData-_fat; iData++){
            uValue=(*_paramVector->_mat)[iData];
            for (int iModel=0; iModel<_nModel; iModel++){
                weight=bspline1d(iModel, uValue, _order, _dk, _nModel, _knots);
                (*scaleVector->_mat)[iModel]+=weight*(*scaleVectorData->_mat)[iData];
            }
        }
        // Compute scaling
        for (int iModel=0; iModel<_nModel; iModel++){
            (*scaleVector->_mat)[iModel]=1.0/sqrt((*scaleVector->_mat)[iModel]);
        }
    }

    return scaleVector;
}

std::shared_ptr<double1DReg> interpBSpline1dDouble::computeCurve(int nSample, std::shared_ptr<double1DReg> controlPoints) {

    // Declaration and allocation
    double sampleValue, weight;
    double oSample = 0.0;
    double fSample = 1.0;
    double dSample = (fSample-oSample)/(nSample);
    axis sampleAxis = axis(nSample, oSample, dSample);
    std::shared_ptr<double1DReg> sampleVector (new double1DReg(sampleAxis)); // Create parameter vector
    for (int iSample=0; iSample<nSample; iSample++){
        (*sampleVector->_mat)[iSample]=oSample+dSample*iSample;
    }

    // Allocate interpolated values (x and y)
    _curveX = std::make_shared<double1DReg>(sampleAxis);
    _curveY = std::make_shared<double1DReg>(sampleAxis);
    _curveX->scale(0.0);
    _curveY->scale(0.0);

    // Compute curve for all parameter values
    for (int iSample=0; iSample<nSample; iSample++){
        sampleValue = (*sampleVector->_mat)[iSample];
        for (int iModel=0; iModel<_nModel; iModel++){
            weight=bspline1d(iModel, sampleValue, _order, _dk, _nModel, _knots);
            (*_curveX->_mat)[iSample]+=weight*(*_xModel->_mat)[iModel]; // Interpolate x-position
            (*_curveY->_mat)[iSample]+=weight*(*controlPoints->_mat)[iModel]; // Interpolate y-position
        }
    }
    return sampleVector;
}

double interpBSpline1dDouble::bspline1d(int iControl, double uValue, int order, double dk, int nControlPoints, std::shared_ptr<double1DReg> knots) const {

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
        std::cout << "**** ERROR: Order not supported. Please choose from 0->3 (included) ****" << std::endl;
        throw std::runtime_error("");
    }
    return weight;
}

// Forward
void interpBSpline1dDouble::forward(const bool add, const std::shared_ptr<double1DReg> model, std::shared_ptr<double1DReg> data) const {

	/* FORWARD: COARSE -> FINE GRID */
    // Data has to be on a regular grid
    // Model can be on an irregular grid
	if (!add) data->scale(0.0);

    // Declare spline weighting
    double weight;
    double uValue;

    // Loop over data (fine sampling grid)
    for (int iData=_fat; iData<_nData-_fat; iData++){

        // Compute parameter value for iData
        uValue=(*_paramVector->_mat)[iData];

        // Loop over model (control points)
        for (int iModel=0; iModel<_nModel; iModel++){
            weight=bspline1d(iModel, uValue, _order, _dk, _nModel, _knots); // Compute spline weight
            (*data->_mat)[iData]+=weight*(*_scaleVector->_mat)[iModel]*(*model->_mat)[iModel]; // Add contribution from control point
        }
    }
}

// Adjoint
void interpBSpline1dDouble::adjoint(const bool add, std::shared_ptr<double1DReg> model, const std::shared_ptr<double1DReg> data) const {

	/* ADJOINT: FINE -> COARSE GRID */
	if (!add) model->scale(0.0);

    // Declare spline weighting
    double weight;
    double uValue;

    // Loop model (control points)
    for (int iModel=0; iModel<_nModel; iModel++){
        // Loop over data
        for (int iData=_fat; iData<_nData-_fat; iData++){
            uValue=(*_paramVector->_mat)[iData];
            weight=bspline1d(iModel, uValue, _order, _dk, _nModel, _knots); // Compute spline weight
            (*model->_mat)[iModel]+=weight*(*_scaleVector->_mat)[iModel]*(*data->_mat)[iData]; // Add contribution from control point
        }
    }
}

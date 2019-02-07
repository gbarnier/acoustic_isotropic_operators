#include <iostream>
#include <time.h>
#include "float1DReg.h"
#include "float2DReg.h"
#include "float3DReg.h"
#include "ioModes.h"
#include <vector>
#include <string>

using namespace SEP;

int main(int argc, char **argv) {

	// Bullshit
	ioModes modes(argc, argv);
	std::shared_ptr <SEP::genericIO> io = modes.getDefaultIO();
	std::shared_ptr <paramObj> par = io->getParamObj();

    // Control points
    int nControlPoints = 3;
    std::shared_ptr<float1DReg> controlPointsY(new float1DReg(nControlPoints));
    std::shared_ptr<float1DReg> controlPointsX(new float1DReg(nControlPoints));

    // Manually fill in control points x and y values
    (*controlPointsX->_mat)[0] = 0;
    (*controlPointsX->_mat)[1] = 1;
    (*controlPointsX->_mat)[2] = 2;
    (*controlPointsY->_mat)[0] = 2;
    (*controlPointsY->_mat)[1] = 3;
    (*controlPointsY->_mat)[2] = 0.5;

    // Basis functions order
    int order = 0;

    // Knot vector
    int nKnots = nControlPoints + order + 1;
    std::shared_ptr<float1DReg> knots(new float1DReg(nKnots));

    // Set first and last knots to be equal to first and last control points
    (*knots->_mat)[0] = (*controlPointsX->_mat)[0];
    (*knots->_mat)[nKnots-1] = (*controlPointsX->_mat)[nControlPoints-1];

    // Compute interval between knots
    float dKnot = ( (*controlPointsX->_mat)[nControlPoints-1]-(*controlPointsX->_mat)[0] )/(nKnots-1);

	// Compute inner knots
    for (int iKnot=1; iKnot<nControlPoints-1; iKnot++){
        (*knots->_mat)[iKnot] = (*knots->_mat)[0] + iKnot*dKnot;
    }

	// Generate the set of points where you want to interpolate
    std::shared_ptr<float1DReg> x(new float1DReg(nControlPoints));
	float dx = 0.2;
	int nx = ((*controlPointsX->_mat)[2]-(*controlPointsX->_mat)[0]) / dx + 1;

	// x-positions of the interpolated points
	for (int ix=0; ix<nx; ix++){
		(*x->_mat)[ix] = (*controlPointsX->_mat)[0] + ix * dx;
	}

	// values of the interpolated points
    std::shared_ptr<float1DReg> y(new float1DReg(nx));

	// Compute interpolation for x vector
	for (int ix=0; ix<nx; ix++){

		// Compute x-position
		float xPos = (*x->_mat)[ix];
		std::cout << "x-position = " << xPos << std::endl;

		// Initialize partial sum
		float sum = 0;
		float weight;
		for (int iControlPoint=0; iControlPoint<nControlPoints-1; iControlPoint++){

			std::cout << "control point nb = " << iControlPoint << std::endl;

			// Compute B-spline basis function for (iControlPoints,order)
			// Basis functions are non-zero only on span [u_i,u_{i+p+1})
			if ((xPos >= (*knots->_mat)[iControlPoint]) && (xPos < (*knots->_mat)[iControlPoint+1])){
				std::cout << "x is between " << (*knots->_mat)[iControlPoint] << "and " << (*knots->_mat)[iControlPoint+1];
				weight = 1;
			} else {
				std::cout << "x is not between " << (*knots->_mat)[iControlPoint] << "and " << (*knots->_mat)[iControlPoint+1];
				weight = 0.0;
			}

			// Add the weighted control point to the partial sum
			sum += weight * (*controlPointsY->_mat)[iControlPoint];

		}
		(*y->_mat)[ix] = sum;
	}


	// Output shits

	// Files for interpolated curve

	// x-coordinates
	std::shared_ptr<SEP::genericRegFile> xFile = io->getRegFile("x",usageOut);
	xFile->setHyper(x->getHyper());
	xFile->writeDescription();
	xFile->writeFloatStream(x);

	// y-coordinates
	std::shared_ptr<SEP::genericRegFile> yFile = io->getRegFile("y",usageOut);
	yFile->setHyper(y->getHyper());
	yFile->writeDescription();
	yFile->writeFloatStream(y);

	return 0;

}

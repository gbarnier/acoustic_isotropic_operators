#include <math.h>
#include "int1DReg.h"
#include "double1DReg.h"
#include "interpSplineLinear.h"

using namespace SEP;

interpSplineLinear1d::interpSplineLinear1d(std::shared_ptr<int1DReg> modelGridPos, axis gridAxis){

	_modelGridPos = modelGridPos; // Position of the "spline nodes" on the grid (must be integer values)
	_nModel=_modelGridPos->getHyper()->getAxis(0).n; // Number of "nodes"
	_gridAxis = gridAxis; // Axis grid
	_nData=_gridAxis.n; // Nb of data points to interpolate
	_n1=_gridAxis.n; // Number of points on axis at fine sampling (= nb of data points)
	_o1=_gridAxis.o; // Axis origin
	_d1=_gridAxis.d; // Spatial sampling
	_nInterval=_nModel-1; // Number of intervals
	_interpFilter = std::make_shared<double1DReg>(_nInterval); // Interpolation filter

	// Compute filter coefficients
	for (int iData=0; iData<_nData; iData++){

		// Compute position of data point
		double d=_o1+(*_modelGridPos->_mat)[iData]*_d1;

		// Compute nodes surrounding the data point
		for (int iNode)

	}



}

void interpSplineLinear1d::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const {

	if (!add) data->scale(0.0);




}

void interpSplineLinear1d::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const {

	if (!add) model->scale(0.0);



}

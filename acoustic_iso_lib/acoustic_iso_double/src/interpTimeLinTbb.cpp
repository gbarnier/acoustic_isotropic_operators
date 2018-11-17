#include <math.h>
#include "double2DReg.h"
#include "interpTimeLinTbb.h"

using namespace SEP;

/****************************** 1D linear interpolation in time *************************/	

interpTimeLinTbb::interpTimeLinTbb(int nts, double dts, double ots, int sub) {
	
	/* Get time sampling parameters */
	_ots = ots;
	_dts = dts;	
	_nts = nts;
	_sub = sub;
	_ntw = (_nts-1) * _sub + 1;
	_dtw = _dts / _sub; 
	_otw = ots;
	_scale = double(_ntw) / double(_nts);
	_scale = 1.0 / sqrt(_scale);
	_timeAxisCoarse = axis(_nts, _ots, _dts);
	_timeAxisFine = axis(_ntw, _otw, _dtw);	

}

void interpTimeLinTbb::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const {

	/* Forward: from coarse grid to fine grid */
	/* Model: coarse grid */
	/* Data: fine grid	*/

	if (!add) data->scale(0.0);

	/* Declare variables */
	int nDevice = model->getHyper()->getAxis(2).n;

  	std::shared_ptr<double2D> d = data->_mat; 
    const std::shared_ptr<double2D> m = model->_mat;

	tbb::parallel_for(
		
		// Argument #1: range
		tbb::blocked_range<int>(0, nDevice),
		
		// Argument #2: body (Lambda function)
		[&](const tbb::blocked_range<int> &r){
		
			for (int iDevice = r.begin(); iDevice != r.end(); iDevice++) {
		
				for (int itw = 0; itw < _ntw-1; itw++) {
					double tw = _otw + itw * _dtw; // Compute time on fine grid
					double	weight = (tw - _ots) / _dts; // Compute number of coarse time samples
					int	indexInf = weight; // Compute floor of number of coarse time samples
					weight = weight - indexInf; // Compute weight
					(*d)[iDevice][itw] += ( (*m)[iDevice][indexInf] * (1.0 - weight) + (*m)[iDevice][indexInf+1] * weight ) * _scale;
				}
				/* Treat the last sample separately */
				(*d)[iDevice][_ntw-1] += _scale * (*m)[iDevice][_nts-1];
			}
		}
	);	
}		

void interpTimeLinTbb::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const {

	/* Adjoint: from fine grid to coarse grid */
	/* Model: coarse grid */
	/* Data: fine grid */

	if (!add) model->scale(0.0);

	/* Declare variables */
	double tw, weight;
	int indexInf; 
	int nDevice = model->getHyper()->getAxis(2).n;
	
  	std::shared_ptr<double2D> d = data->_mat; 
    const std::shared_ptr<double2D> m = model->_mat;

	tbb::parallel_for(

		// Argument #1: range
		tbb::blocked_range<int>(0, nDevice),

		// Argument #2: body (Lambda function)
		[&](const tbb::blocked_range<int> &r){

			for (int iDevice = r.begin(); iDevice != r.end(); iDevice++){
				for (int itw = 0; itw < _ntw-1; itw++){
					double tw = _otw + itw * _dtw; // Compute time on fine grid
					double weight = (tw - _ots) / _dts; // Compute number of coarse time samples
					int indexInf = weight; // Compute floor of number of coarse time samples
					weight = weight - indexInf; // Compute weight
					(*m)[iDevice][indexInf]   += _scale * (*d)[iDevice][itw] * (1.0 - weight);
					(*m)[iDevice][indexInf+1] += _scale * (*d)[iDevice][itw] * weight;
				}
			/* Treat the last sample separately */
			(*m)[iDevice][_nts-1] += _scale * (*d)[iDevice][_ntw-1];
			}
		}
	);
}



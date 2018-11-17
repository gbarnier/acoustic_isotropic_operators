#include <double2DReg.h>
#include "secondTimeDerivative.h"
#include <math.h>

using namespace SEP;

/****************************** 1D linear interpolation in time *************************/	

secondTimeDerivative::secondTimeDerivative(int nt, double dt) {
	_nt = nt;
	_dt2 = 1.0 / (dt*dt);	
}

void secondTimeDerivative::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double2DReg> data) const
{

	if (!add) data->scale(0.0);

	for (int iDevice = 0; iDevice < model->getHyper()->getAxis(2).n; iDevice++) {		
		(*data->_mat)[iDevice][0] += _dt2 * ( (*model->_mat)[iDevice][1] - 2.0 * (*model->_mat)[iDevice][0] );
		for (int it = 1; it < _nt-1; it++) {
			(*data->_mat)[iDevice][it] += _dt2 * ( (*model->_mat)[iDevice][it+1] - 2.0 * (*model->_mat)[iDevice][it] + (*model->_mat)[iDevice][it-1] );		
		}
		(*data->_mat)[iDevice][_nt-1] += _dt2 * ( (*model->_mat)[iDevice][_nt-2] - 2.0 * (*model->_mat)[iDevice][_nt-1] );	
	}
}

void secondTimeDerivative::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double2DReg> data) const{

	if (!add) model->scale(0.0);

	for (int iDevice = 0; iDevice < data->getHyper()->getAxis(2).n; iDevice++) {		
		(*model->_mat)[iDevice][0] += _dt2 * ( (*data->_mat)[iDevice][1] - 2.0 * (*data->_mat)[iDevice][0] );
		for (int it = 1; it < _nt-1; it++) {
			(*model->_mat)[iDevice][it] += _dt2 * ( (*data->_mat)[iDevice][it+1] - 2.0 * (*data->_mat)[iDevice][it] + (*data->_mat)[iDevice][it-1] );		
		}
		(*model->_mat)[iDevice][_nt-1] += _dt2 * ( (*data->_mat)[iDevice][_nt-2] - 2.0 * (*data->_mat)[iDevice][_nt-1] );	
	}
}
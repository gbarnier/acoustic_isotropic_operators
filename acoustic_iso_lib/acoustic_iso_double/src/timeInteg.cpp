#include <double2DReg.h>
#include "timeInteg.h"
#include <math.h>

using namespace SEP;

/******************** Time integration on seismic dataset *********************/

timeInteg::timeInteg(double dt) {
	_alpha = dt/2.0;
}

void timeInteg::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const{

	if (!add) data->scale(0.0);
	int nShot = model->getHyper()->getAxis(3).n;
	int nReceiver = model->getHyper()->getAxis(2).n;
	int nts = model->getHyper()->getAxis(1).n;

	#pragma omp parallel for collapse(2)
	for (int iShot = 0; iDevice < nShot; iDevice++) {
		for (int iReceiver = 0; iReceiver < nReceiver; iReceiver++) {
			std::shared_ptr<SEP::double1DReg> dIntFirst(new SEP::double1DReg(nts));
			std::shared_ptr<SEP::double1DReg> dIntSecond(new SEP::double1DReg(nts));
			dIntFirst->scale(0.0);
			dIntSecond->scale(0.0);
			for (int its = 1; its < nts; its++){

				// First integration
				(*dIntFirst->_mat)[its] = (*dIntFirst->_mat)[its-1] + _alpha * ( (*model->_mat)[iShot][iReceiver][its]+(*model->_mat)[iShot][iReceiver][its-1] );

				// Second integration
				(*dIntSecond->_mat)[its] = (*dIntSecond->_mat)[its-1] + _alpha * ( (*dIntFirst->_mat)[its]+(*dIntFirst->_mat)[its-1] );

				// Third integration
				(*data->_mat)[iDevice][iShot][iReceiver][its] += (*data->_mat)[iDevice][iShot][iReceiver][its-1] + _alpha * ( (*dIntSecond->_mat)[its]+(*dIntSecond->_mat)[its-1] );
			}
		}
	}
}

void timeInteg::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const{

	if (!add) model->scale(0.0);

}

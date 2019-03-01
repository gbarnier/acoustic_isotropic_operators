#include <string>
#include <float2DReg.h>
#include <iostream>
#include "interpRbf1d.h"
#include <omp.h>
#include <vector>
#include <math.h>

// Constructor
interpRbf1d::interpRbf1d(float epsilon, std::shared_ptr<float1DReg> zModel, axis zAxis, int scaling, int fat){

	_fat = fat;
	_nzModel=zModel->getHyper()->getAxis(1).n;
	_zModel=zModel;
	_zAxis=zAxis;
	_nzData=_zAxis.n;
	_zData=std::make_shared<float1DReg>(_zAxis);
	for (int izData=0; izData<_nzData; izData++){
		(*_zData->_mat)[izData]=_zAxis.o+izData*_zAxis.d;
	}
	_epsilon2=epsilon*epsilon;
	_scaling=scaling;
	computeScaleVector();

}

void interpRbf1d::computeScaleVector(){

	std::shared_ptr<float1DReg> scaleVectorData = _zData->clone();
	_scaleVector = _zModel->clone();
	scaleVectorData->scale(0.0);

	#pragma omp parallel for
	for (int izModel=0; izModel<_nzModel; izModel++){
		(*_scaleVector->_mat)[izModel]=1.0;
	}

	if (_scaling == 1){

		// Apply forward
		#pragma omp parallel for collapse(2)
		for (int izData=_fat; izData<_nzData-_fat; izData++){
			for (int izModel=0; izModel<_nzModel; izModel++){
				float arg = (*_zData->_mat)[izData]-(*_zModel->_mat)[izModel];
				arg=arg*arg;
				arg=-1.0*_epsilon2*arg;
				(*scaleVectorData->_mat)[izData] += exp(arg)* (*_scaleVector->_mat)[izModel];
			}
		}

		// Set scaleVector model to 0
		_scaleVector->scale(0.0);

		// Apply adjoint
		for (int izData=_fat; izData<_nzData-_fat; izData++){
			for (int izModel=0; izModel<_nzModel; izModel++){
				float arg = (*_zData->_mat)[izData]-(*_zModel->_mat)[izModel];
				arg=arg*arg;
				arg=-1.0*_epsilon2*arg;
				(*_scaleVector->_mat)[izModel] += exp(arg)* (*scaleVectorData->_mat)[izData];
			}
		}

		// Compute model scaling
		for (int izModel=0; izModel<_nzModel; izModel++){
			(*_scaleVector->_mat)[izModel]=1.0/sqrt((*_scaleVector->_mat)[izModel]);
		}
	}
}

// Forward
void interpRbf1d::forward(const bool add, const std::shared_ptr<float1DReg> model, std::shared_ptr<float1DReg> data) const {

	if (!add) data->scale(0.0);

	#pragma omp parallel for
	for (int izData=_fat; izData<_nzData-_fat; izData++){
		for (int izModel=0; izModel<_nzModel; izModel++){
			float arg = (*_zData->_mat)[izData]-(*_zModel->_mat)[izModel];
			arg=arg*arg;
			arg=-1.0*_epsilon2*arg;
			(*data->_mat)[izData] += exp(arg)*(*_scaleVector->_mat)[izModel]*(*model->_mat)[izModel];
		}
	}
}

// Adjoint
void interpRbf1d::adjoint(const bool add, std::shared_ptr<float1DReg> model, const std::shared_ptr<float1DReg> data) const {

    if (!add) model->scale(0.0);

	#pragma omp parallel for
	for (int izModel=0; izModel<_nzModel; izModel++){
		for (int izData=_fat; izData<_nzData-_fat; izData++){
			float arg = (*_zData->_mat)[izData]-(*_zModel->_mat)[izModel];
			arg=arg*arg;
			arg=-1.0*_epsilon2*arg;
			(*model->_mat)[izModel] += exp(arg)*(*_scaleVector->_mat)[izModel]*(*data->_mat)[izData];
		}
	}
}

#include <vector>
#include <ctime>
#include "nonlinearPropGpu.h"

nonlinearPropGpu::nonlinearPropGpu(std::shared_ptr<SEP::float2DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){
	_fdParam = std::make_shared<fdParam>(vel, par);
	_timeInterp = std::make_shared<interpTimeLinTbb>(_fdParam->_nts, _fdParam->_dts, _fdParam->_ots, _fdParam->_sub);
	setAllWavefields(par->getInt("saveWavefield", 0));
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;

	// Initialize GPU
	initNonlinearGpu(_fdParam->_dz, _fdParam->_dx, _fdParam->_nz, _fdParam->_nx, _fdParam->_nts, _fdParam->_dts, _fdParam->_sub, _fdParam->_minPad, _fdParam->_blockSize, _fdParam->_alphaCos, _nGpu, _iGpuId, iGpuAlloc);
}

void nonlinearPropGpu::setAllWavefields(int wavefieldFlag){
	_wavefield = setWavefield(wavefieldFlag);
}

bool nonlinearPropGpu::checkParfileConsistency(const std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float2DReg> data) const{

	if (_fdParam->checkParfileConsistencyTime(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam->checkParfileConsistencyTime(model,1, "Model file") != true) {return false;}; // Check model time axis

	return true;
}

void nonlinearPropGpu::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const {

	if (!add) data->scale(0.0);

 	std::clock_t start;
    float duration;

	/* Allocation */
	std::shared_ptr<float2DReg> modelRegDts(new float2DReg(_fdParam->_nts, _nSourcesReg));
	std::shared_ptr<float2DReg> modelRegDtw(new float2DReg(_fdParam->_ntw, _nSourcesReg));
	std::shared_ptr<float2DReg> dataRegDtw(new float2DReg(_fdParam->_ntw, _nReceiversReg));
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam->_nts, _nReceiversReg));

	/* Interpolate model (seismic source) to regular grid */
	_sources->adjoint(false, modelRegDts, model);

	/* Scale model by dtw^2 * vel^2 * dSurface */
	scaleSeismicSource(_sources, modelRegDts, _fdParam);

	/* Interpolate to fine time-sampling */
	_timeInterp->forward(false, modelRegDts, modelRegDtw);

	/* Propagate */
	if (_saveWavefield == 0) {
		propShotsFwdGpu(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);
    } else {
		propShotsFwdGpuWavefield(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);
	}

	/* Interpolate to irregular grid */
	_receivers->forward(true, dataRegDts, data);

}

void nonlinearPropGpu::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam->_nts, _nReceiversReg));
	std::shared_ptr<float2DReg> modelRegDtw(new float2DReg(_fdParam->_ntw, _nSourcesReg));
	std::shared_ptr<float2DReg> modelRegDts(new float2DReg(_fdParam->_nts, _nSourcesReg));

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Propagate */
	if (_saveWavefield == 0) {
		propShotsAdjGpu(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);
	} else {
		propShotsAdjGpuWavefield(modelRegDtw->getVals(), dataRegDts->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _wavefield->getVals(), _iGpu, _iGpuId);
	}

	/* Scale adjoint wavefield */
	if (_saveWavefield == 1){
		#pragma omp parallel for
		for (int its=0; its<_fdParam->_nts; its++){
			for (int ix=0; ix<_fdParam->_nx; ix++){
				for (int iz=0; iz<_fdParam->_nz; iz++){
					(*_wavefield->_mat)[its][ix][iz] *= _fdParam->_dtw*_fdParam->_dtw*(*_fdParam->_vel->_mat)[ix][iz]*(*_fdParam->_vel->_mat)[ix][iz];
				}
			}
		}
	}

	/* Interpolate to coarse time-sampling */
	_timeInterp->adjoint(false, modelRegDts, modelRegDtw);

	/* Scale model by dtw^2 * vel^2 * dSurface */
	scaleSeismicSource(_sources, modelRegDts, _fdParam);

	/* Interpolate to irregular grid */
	_sources->forward(true, modelRegDts, model);

}

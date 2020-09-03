#include "tomoExtGpu.h"

tomoExtGpu::tomoExtGpu(std::shared_ptr<SEP::float2DReg> vel, std::shared_ptr<paramObj> par, std::shared_ptr<SEP::float3DReg> reflectivityExt, int nGpu, int iGpu, int iGpuId, int iGpuAlloc){

	// Finite-difference parameters
	_fdParam = std::make_shared<fdParam>(vel, par);
	_timeInterp = std::make_shared<interpTimeLinTbb>(_fdParam->_nts, _fdParam->_dts, _fdParam->_ots, _fdParam->_sub);
	_secTimeDer = std::make_shared<secondTimeDerivative>(_fdParam->_nts, _fdParam->_dts);
	_reflectivityExt = reflectivityExt;
	_leg1 = par->getInt("leg1", 1);
	_leg2 = par->getInt("leg2", 1);
	_iGpu = iGpu;
	_nGpu = nGpu;
	_iGpuId = iGpuId;

	setAllWavefields(par->getInt("saveWavefield", 0));

	// Initialize GPU
	initTomoExtGpu(_fdParam->_dz, _fdParam->_dx, _fdParam->_nz, _fdParam->_nx, _fdParam->_nts, _fdParam->_dts, _fdParam->_sub, _fdParam->_minPad, _fdParam->_blockSize, _fdParam->_alphaCos, _fdParam->_nExt, _leg1, _leg2, _nGpu, _iGpuId, iGpuAlloc);
}

bool tomoExtGpu::checkParfileConsistency(const std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float2DReg> data) const {
	if (_fdParam->checkParfileConsistencyTime(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam->checkParfileConsistencyTime(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam->checkParfileConsistencySpace(model, "Model file") != true) {return false;}; // Check model space axes
	if (_fdParam->checkParfileConsistencySpace(_reflectivityExt, "Reflectivity file") != true) {return false;}; // Check extended reflectivity axes
	return true;
}

void tomoExtGpu::setAllWavefields(int wavefieldFlag){
	// -> The allocations are done here
	_srcWavefield = setWavefield(wavefieldFlag);
	_secWavefield1 = setWavefield(wavefieldFlag);
	_secWavefield2 = setWavefield(wavefieldFlag);
}

void tomoExtGpu::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const {

	if (!add) data->scale(0.0);
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam->_nts, _nReceiversReg));

	/* Tomo extended forward */

	// No free surface
	if (_fdParam->_freeSurface == 0){

		if (_fdParam->_extension == "time"){
			tomoTimeShotsFwdGpu(model->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		}
		if (_fdParam->_extension == "offset"){
			tomoOffsetShotsFwdGpu(model->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		}
	// Free surface
	} else {
		if (_fdParam->_extension == "time"){
			tomoTimeShotsFwdFsGpu(model->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		}
		if (_fdParam->_extension == "offset"){
			tomoOffsetShotsFwdFsGpu(model->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		}
	}
	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);
}

void tomoExtGpu::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam->_nts, _nReceiversReg));
	std::shared_ptr<float2DReg> modelTemp = model->clone(); // We need to create a temporary model for "add"
	modelTemp->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Tomo extended adjoint */

	// No free surface
	if (_fdParam->_freeSurface == 0){

		if (_fdParam->_extension == "time"){
			tomoTimeShotsAdjGpu(modelTemp->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		}
		if (_fdParam->_extension == "offset"){
			tomoOffsetShotsAdjGpu(modelTemp->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		}
	// Free surface		
	} else {
		if (_fdParam->_extension == "time"){
			tomoTimeShotsAdjFsGpu(modelTemp->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		}
		if (_fdParam->_extension == "offset"){
			tomoOffsetShotsAdjFsGpu(modelTemp->getVals(), dataRegDts->getVals(), _reflectivityExt->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _iGpuId, _saveWavefield);
		}
	}

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);
}

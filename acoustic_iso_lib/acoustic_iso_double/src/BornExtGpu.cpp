#include "BornExtGpu.h"

BornExtGpu::BornExtGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu) {

	// Finite-difference parameters
	_fdParam = std::make_shared<fdParam>(vel, par);
	_timeInterp = std::make_shared<interpTimeLinTbb>(_fdParam->_nts, _fdParam->_dts, _fdParam->_ots, _fdParam->_sub);
	_secTimeDer = std::make_shared<secondTimeDerivative>(_fdParam->_nts, _fdParam->_dts);
	// setWavefield(par->getInt("saveWavefield"));
	_iGpu = iGpu;
	_nGpu = nGpu;
	setAllWavefields(par->getInt("saveWavefield", 0));

	// Initialize GPU
	initBornExtGpu(_fdParam->_dz, _fdParam->_dx, _fdParam->_nz, _fdParam->_nx, _fdParam->_nts, _fdParam->_dts, _fdParam->_sub, _fdParam->_minPad, _fdParam->_blockSize, _fdParam->_alphaCos, _fdParam->_nExt, _nGpu, _iGpu);
}

bool BornExtGpu::checkParfileConsistency(const std::shared_ptr<SEP::double3DReg> model, const std::shared_ptr<SEP::double2DReg> data) const {
	if (_fdParam->checkParfileConsistencyTime(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam->checkParfileConsistencyTime(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam->checkParfileConsistencySpace(model, "Model file") != true) {return false;}; // Check model space axes
	return true;

}

void BornExtGpu::setAllWavefields(int wavefieldFlag){
	_srcWavefield = setWavefield(wavefieldFlag);
	_secWavefield = setWavefield(wavefieldFlag);
}

void BornExtGpu::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double2DReg> data) const {
	if (!add) data->scale(0.0);
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam->_nts, _nReceiversReg));

	/* Time-lags extension */
	if (_fdParam->_extension == "time") {
		if (_saveWavefield == 0){
			BornTimeShotsFwdGpu(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
		} else {
			BornTimeShotsFwdGpuWavefield(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
		}
	/* Subsurface offsets extension */
	} else {
		if (_saveWavefield == 0){
			BornOffsetShotsFwdGpu(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
		} else {
			BornOffsetShotsFwdGpuWavefield(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
		}
	}

	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);

}

void BornExtGpu::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double2DReg> data) const {

	if (!add) model->scale(0.0);
	std::shared_ptr<double2DReg> dataRegDts(new double2DReg(_fdParam->_nts, _nReceiversReg));
	std::shared_ptr<double3DReg> modelTemp = model->clone(); // We need to create a temporary model for "add"
	modelTemp->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Launch time-extended Born adjoint */
	if (_fdParam->_extension == "time") {
		if (_saveWavefield == 0){
			BornTimeShotsAdjGpu(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
		} else {
			BornTimeShotsAdjGpuWavefield(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
		}
	}

	/* Launch offset-extended Born adjoint */
	if (_fdParam->_extension == "offset") {
		if (_saveWavefield == 0){
			BornOffsetShotsAdjGpu(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
		}
		else {
			BornOffsetShotsAdjGpuWavefield(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
		}
	}

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);
}

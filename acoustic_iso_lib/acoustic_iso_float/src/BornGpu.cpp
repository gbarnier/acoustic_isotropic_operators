#include "BornGpu.h"

BornGpu::BornGpu(std::shared_ptr<SEP::float2DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu) {

	_fdParam = std::make_shared<fdParam>(vel, par); // Fd parameter object
	_timeInterp = std::make_shared<interpTimeLinTbb>(_fdParam->_nts, _fdParam->_dts, _fdParam->_ots, _fdParam->_sub); // Time interpolation object
	_secTimeDer = std::make_shared<secondTimeDerivative>(_fdParam->_nts, _fdParam->_dts); // Second time derivative object
	_iGpu = iGpu; // Gpu number
	_nGpu = nGpu; // Number of requested GPUs
	setAllWavefields(par->getInt("saveWavefield", 0));

	// Initialize GPU
	initBornGpu(_fdParam->_dz, _fdParam->_dx, _fdParam->_nz, _fdParam->_nx, _fdParam->_nts, _fdParam->_dts, _fdParam->_sub, _fdParam->_minPad, _fdParam->_blockSize, _fdParam->_alphaCos, _nGpu, _iGpu);
}

bool BornGpu::checkParfileConsistency(const std::shared_ptr<SEP::float2DReg> model, const std::shared_ptr<SEP::float2DReg> data) const {
	if (_fdParam->checkParfileConsistencyTime(data, 1, "Data file") != true) {return false;} // Check data time axis
	if (_fdParam->checkParfileConsistencyTime(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check wavelet time axis
	if (_fdParam->checkParfileConsistencySpace(model, "Model file") != true) {return false;}; // Check model space axes
	return true;
}

void BornGpu::setAllWavefields(int wavefieldFlag){
	_srcWavefield = setWavefield(wavefieldFlag);
	_secWavefield = setWavefield(wavefieldFlag);
}

void BornGpu::forward(const bool add, const std::shared_ptr<float2DReg> model, std::shared_ptr<float2DReg> data) const {

	if (!add) data->scale(0.0);

	/* Allocation */
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam->_nts, _nReceiversReg));

	/* Launch Born forward */
	if (_saveWavefield == 0){
		BornShotsFwdGpu(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
	} else {
		BornShotsFwdGpuWavefield(model->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
	}

	/* Interpolate data to irregular grid */
	_receivers->forward(true, dataRegDts, data);

}
void BornGpu::adjoint(const bool add, std::shared_ptr<float2DReg> model, const std::shared_ptr<float2DReg> data) const {

	if (!add) model->scale(0.0);

	/* Allocation */
	std::shared_ptr<float2DReg> dataRegDts(new float2DReg(_fdParam->_nts, _nReceiversReg));
	std::shared_ptr<float2DReg> modelTemp = model->clone();
	modelTemp->scale(0.0);

	/* Interpolate data to regular grid */
	_receivers->adjoint(false, dataRegDts, data);

	/* Launch Born adjoint */
	if (_saveWavefield == 0){
		BornShotsAdjGpu(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
	} else {
		BornShotsAdjGpuWavefield(modelTemp->getVals(), dataRegDts->getVals(), _sourcesSignalsRegDtwDt2->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield->getVals(), _iGpu);
	}

	/* Update model */
	model->scaleAdd(modelTemp, 1.0, 1.0);

}

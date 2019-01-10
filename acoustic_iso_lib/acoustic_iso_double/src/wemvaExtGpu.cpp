#include "wemvaExtGpu.h"

// Overloaded constructor
wemvaExtGpu::wemvaExtGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, int nGpu, int iGpu) {

	// Finite-difference parameters
	_fdParam = std::make_shared<fdParam>(vel, par);
	_timeInterp = std::make_shared<interpTimeLinTbb>(_fdParam->_nts, _fdParam->_dts, _fdParam->_ots, _fdParam->_sub);
	_secTimeDer = std::make_shared<secondTimeDerivative>(_fdParam->_nts, _fdParam->_dts);

    // Operator parameters
	_leg1 = par->getInt("leg1", 1);
	_leg2 = par->getInt("leg2", 1);
	_iGpu = iGpu;
	_nGpu = nGpu;

	setAllWavefields(par->getInt("saveWavefield", 0));

	// Initialize GPU
	initWemvaExtGpu(_fdParam->_dz, _fdParam->_dx, _fdParam->_nz, _fdParam->_nx, _fdParam->_nts, _fdParam->_dts, _fdParam->_sub, _fdParam->_minPad, _fdParam->_blockSize, _fdParam->_alphaCos, _fdParam->_nExt, _leg1, _leg2, _nGpu, _iGpu);
}

// Sources setup
void wemvaExtGpu::setSources(std::shared_ptr<deviceGpu> sourcesDevices, std::shared_ptr<SEP::double2DReg> sourcesSignals){

	// Set source devices
	_sources = sourcesDevices;
	_nSourcesReg = _sources->getNDeviceReg();
	_sourcesPositionReg = _sources->getRegPosUnique();

	// Set source signals
	_sourcesSignals = sourcesSignals;
	_sourcesSignalsRegDts = std::make_shared<SEP::double2DReg>(_fdParam->_nts, _nSourcesReg);
	_sourcesSignalsRegDtsDt2 = std::make_shared<SEP::double2DReg>(_fdParam->_nts, _nSourcesReg);
	_sourcesSignalsRegDtwDt2 = std::make_shared<SEP::double2DReg>(_fdParam->_ntw, _nSourcesReg);
	_sourcesSignalsRegDtw = std::make_shared<SEP::double2DReg>(_fdParam->_ntw, _nSourcesReg);
	_sources->adjoint(false, _sourcesSignalsRegDts, _sourcesSignals);
	_secTimeDer->forward(false, _sourcesSignalsRegDts, _sourcesSignalsRegDtsDt2);
	scaleSeismicSource(_sources, _sourcesSignalsRegDtsDt2, _fdParam);
	scaleSeismicSource(_sources, _sourcesSignalsRegDts, _fdParam);
	_timeInterp->forward(false, _sourcesSignalsRegDtsDt2, _sourcesSignalsRegDtwDt2);
	_timeInterp->forward(false, _sourcesSignalsRegDts, _sourcesSignalsRegDtw);

}

// Receivers setup
void wemvaExtGpu::setReceivers(std::shared_ptr<deviceGpu> receiversDevices, std::shared_ptr<SEP::double2DReg> receiversSignals){

	// Set receiver devices
	_receivers = receiversDevices;
	_nReceiversReg = _receivers->getNDeviceReg();
	_receiversPositionReg = _receivers->getRegPosUnique();

	// Set receiver signals (Born data)
	_receiversSignals = receiversSignals;
	_receiversSignalsRegDts = std::make_shared<SEP::double2DReg>(_fdParam->_nts, _nReceiversReg);
	_receivers->adjoint(false, _receiversSignalsRegDts, _receiversSignals);

}

// Acquisition setup + quality check with parfile
void wemvaExtGpu::setAcquisition(std::shared_ptr<deviceGpu> sources, std::shared_ptr<SEP::double2DReg> sourcesSignals, std::shared_ptr<deviceGpu> receivers, std::shared_ptr<SEP::double2DReg> receiversSignals, const std::shared_ptr<SEP::double2DReg> model, const std::shared_ptr<double3DReg> data){

	setSources(sources, sourcesSignals);
	setReceivers(receivers, receiversSignals);
	this->setDomainRange(model, data);
	assert(checkParfileConsistency(model, data));

}

// QC with parfile
bool wemvaExtGpu::checkParfileConsistency(const std::shared_ptr<SEP::double2DReg> model, const std::shared_ptr<SEP::double3DReg> data) const {
	if (_fdParam->checkParfileConsistencySpace(data, "Extended image file") != true) {return false;} // Check data time axis
	if (_fdParam->checkParfileConsistencySpace(model, "Model file") != true) {return false;}; // Check model space axes
	if (_fdParam->checkParfileConsistencyTime(_sourcesSignals, 1, "Seismic source file") != true) {return false;}; // Check model space axes
	if (_fdParam->checkParfileConsistencyTime(_receiversSignals, 1, "Wemva data file") != true) {return false;}; // Check model space axes
	return true;
}

// Scaling before propagation
void wemvaExtGpu::scaleSeismicSource(const std::shared_ptr<deviceGpu> seismicSource, std::shared_ptr<SEP::double2DReg> signal, const std::shared_ptr<fdParam> parObj){

	std::shared_ptr<double2D> sig = signal->_mat;
	double *v = _fdParam->_vel->getVals();
	int *pos = seismicSource->getRegPosUnique();

	#pragma omp parallel for
	for (int iGridPoint = 0; iGridPoint < seismicSource->getNDeviceReg(); iGridPoint++){
		double scale = _fdParam->_dtw * _fdParam->_dtw * v[pos[iGridPoint]]*v[pos[iGridPoint]];
		for (int it = 0; it < signal->getHyper()->getAxis(1).n; it++){
			(*sig)[iGridPoint][it] = (*sig)[iGridPoint][it] * scale;
		}
	}
}

std::shared_ptr<SEP::double3DReg> wemvaExtGpu::setWavefield(int wavefieldFlag){

	_saveWavefield = wavefieldFlag;

	std::shared_ptr<double3DReg> wavefield;
	if (wavefieldFlag == 1) {
		wavefield = std::make_shared<double3DReg>(_fdParam->_zAxis, _fdParam->_xAxis, _fdParam->_timeAxisCoarse);
		unsigned long long int wavefieldSize = _fdParam->_zAxis.n * _fdParam->_xAxis.n;
		wavefieldSize *= _fdParam->_nts*sizeof(double);
		memset(wavefield->getVals(), 0, wavefieldSize);
		return wavefield;
	}
	else {
		wavefield = std::make_shared<double3DReg>(1, 1, 1);
		unsigned long long int wavefieldSize = 1*sizeof(double);
		memset(wavefield->getVals(), 0, wavefieldSize);
		return wavefield;
	}
}

void wemvaExtGpu::setAllWavefields(int wavefieldFlag){
	_srcWavefield = setWavefield(wavefieldFlag);
	_secWavefield1 = setWavefield(wavefieldFlag);
	_secWavefield2 = setWavefield(wavefieldFlag);
}

void wemvaExtGpu::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) data->scale(0.0);

	std::shared_ptr<double3DReg> dataTemp = data->clone();

	// Wemva forward
	wemvaExtShotsFwdGpu(model->getVals(), dataTemp->getVals(), _sourcesSignalsRegDtw->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversSignalsRegDts->getVals(), _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _saveWavefield, _fdParam->_extension);

	/* Update data (extended image) */
	data->scaleAdd(dataTemp, 1.0, 1.0);

}

void wemvaExtGpu::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double3DReg> data) const {

	if (!add) model->scale(0.0);
	std::shared_ptr<double2DReg> modelTemp = model->clone(); // We need to create a temporary model for "add"
	modelTemp->scale(0.0);

	// Wemva adjoint
	wemvaExtShotsAdjGpu(modelTemp->getVals(), data->getVals(), _sourcesSignalsRegDtw->getVals(), _sourcesPositionReg, _nSourcesReg, _receiversSignalsRegDts->getVals(), _receiversPositionReg, _nReceiversReg, _srcWavefield->getVals(), _secWavefield1->getVals(), _secWavefield2->getVals(), _iGpu, _saveWavefield, _fdParam->_extension);

	// Update model
	model->scaleAdd(modelTemp, 1.0, 1.0);

}

// Sources setup for Nonlinear modeling
template <class V1, class V2>
void seismicOperator2D <V1, V2>::setSources(std::shared_ptr<deviceGpu> sources){
	_sources = sources;
	_nSourcesReg = _sources->getNDeviceReg();
	_sourcesPositionReg = _sources->getRegPosUnique();
}

// Sources setup for Born and Tomo
template <class V1, class V2>
void seismicOperator2D <V1, V2>::setSources(std::shared_ptr<deviceGpu> sourcesDevices, std::shared_ptr<V2> sourcesSignals){

	// Set source devices
	_sources = sourcesDevices;
	_nSourcesReg = _sources->getNDeviceReg();
	_sourcesPositionReg = _sources->getRegPosUnique();

	// Set source signals
	_sourcesSignals = sourcesSignals->clone(); // Source signal read from the input file (raw)
	_sourcesSignalsRegDts = std::make_shared<V2>(_fdParam->_nts, _nSourcesReg); // Source signal interpolated to the regular grid
	_sourcesSignalsRegDtsDt2 = std::make_shared<V2>(_fdParam->_nts, _nSourcesReg); // Source signal with second-order time derivative
	_sourcesSignalsRegDtwDt2 = std::make_shared<V2>(_fdParam->_ntw, _nSourcesReg); // Source signal with second-order time derivative on fine time-sampling grid
	_sourcesSignalsRegDtw = std::make_shared<V2>(_fdParam->_ntw, _nSourcesReg); // Source signal on fine time-sampling grid

	// Interpolate spatially to regular grid
	_sources->adjoint(false, _sourcesSignalsRegDts, _sourcesSignals); // Interpolate sources signals to regular grid

	// Apply second time derivative to sources signals
	_secTimeDer->forward(false, _sourcesSignalsRegDts, _sourcesSignalsRegDtsDt2);

	// Scale seismic source
	scaleSeismicSource(_sources, _sourcesSignalsRegDtsDt2, _fdParam); // Scale sources signals by dtw^2 * vel^2
	scaleSeismicSource(_sources, _sourcesSignalsRegDts, _fdParam); // Scale sources signals by dtw^2 * vel^2

	// Interpolate to fine time-sampling
	_timeInterp->forward(false, _sourcesSignalsRegDtsDt2, _sourcesSignalsRegDtwDt2); // Interpolate sources signals to fine time-sampling
	_timeInterp->forward(false, _sourcesSignalsRegDts, _sourcesSignalsRegDtw); // Interpolate sources signals to fine time-sampling

}

// Receivers setup for Nonlinear modeling, Born and Tomo
template <class V1, class V2>
void seismicOperator2D <V1, V2>::setReceivers(std::shared_ptr<deviceGpu> receivers){
	_receivers = receivers;
	_nReceiversReg = _receivers->getNDeviceReg();
	_receiversPositionReg = _receivers->getRegPosUnique();
}

// Set acquisiton for Nonlinear modeling
template <class V1, class V2>
void seismicOperator2D <V1, V2>::setAcquisition(std::shared_ptr<deviceGpu> sources, std::shared_ptr<deviceGpu> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data){
	setSources(sources);
	setReceivers(receivers);
	this->setDomainRange(model, data);
	if( not checkParfileConsistency(model, data)){
		throw std::runtime_error("");
	};
}

// Set acquisiton for Born and Tomo
template <class V1, class V2>
void seismicOperator2D <V1, V2>::setAcquisition(std::shared_ptr<deviceGpu> sources, std::shared_ptr<V2> sourcesSignals, std::shared_ptr<deviceGpu> receivers, const std::shared_ptr<V1> model, const std::shared_ptr<V2> data){
	setSources(sources, sourcesSignals);
	setReceivers(receivers);
	this->setDomainRange(model, data);
	if( not checkParfileConsistency(model, data)){
		throw std::runtime_error("");
	};
}

// Scale seismic source
template <class V1, class V2>
void seismicOperator2D <V1, V2>::scaleSeismicSource(const std::shared_ptr<deviceGpu> seismicSource, std::shared_ptr<V2> signal, const std::shared_ptr<fdParam> parObj) const{

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

// Wavefield setup
template <class V1, class V2>
std::shared_ptr<SEP::double3DReg> seismicOperator2D <V1, V2>:: setWavefield(int wavefieldFlag){

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

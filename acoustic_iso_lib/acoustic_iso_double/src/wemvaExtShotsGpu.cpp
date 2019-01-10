#include <vector>
#include <omp.h>
#include "wemvaExtShotsGpu.h"
#include "wemvaExtGpu.h"

/* Constructor */
wemvaExtShotsGpu::wemvaExtShotsGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<SEP::double2DReg>> sourcesSignalsVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector, std::vector<std::shared_ptr<SEP::double2DReg>> receiversSignalsVector) {

	// Setup parameters
	_par = par;
	_vel = vel;
	_nShot = par->getInt("nShot");
	_nGpu = par->getInt("nGpu");
	_info = par->getInt("info", 0);
	_deviceNumberInfo = par->getInt("deviceNumberInfo", 0);
	// assert(getGpuInfo(_nGpu, _info, _deviceNumberInfo)); // Get info on GPU cluster and check that there are enough available GPUs
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_wavefieldShotNumber = _par->getInt("wavefieldShotNumber", 0);
	_sourcesVector = sourcesVector;
	_receiversVector = receiversVector;
	_sourcesSignalsVector = sourcesSignalsVector;
	_receiversSignalsVector = receiversSignalsVector;

}

/* Forward */
void wemvaExtShotsGpu::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) data->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignalsVector.size() == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal = 0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1){constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vector of wemvaExtGpu objects (as many as the number of GPUs )
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3)));
	std::vector<std::shared_ptr<double3DReg>> dataSliceVector;
	std::vector<std::shared_ptr<wemvaExtGpu>> wemvaExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended wemva object
		std::shared_ptr<wemvaExtGpu> wemvaExtGpuObject(new wemvaExtGpu(_vel, _par, _nGpu, iGpu));
		wemvaExtGpuObjectVector.push_back(wemvaExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			wemvaExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device for that specific GPU number
        allocateWemvaExtShotsGpu(wemvaExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, wemvaExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu);

		// Create data slice (extended image) for this GPU number
		std::shared_ptr<SEP::double3DReg> dataSlice(new SEP::double3DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch Wemva forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], _receiversSignalsVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], _receiversSignalsVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], _receiversSignalsVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], _receiversSignalsVector[iShot], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		wemvaExtGpuObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		wemvaExtGpuObjectVector[iGpu]->forward(false, model, dataSliceVector[iGpu]);

		// Stack data
		#pragma omp parallel for
		for (int iExt=0; iExt<hyperDataSlice->getAxis(3).n; iExt++){
			for (int ix=0; ix<hyperDataSlice->getAxis(2).n; ix++){
                for (int iz=0; iz<hyperDataSlice->getAxis(1).n; iz++){
                    (*data->_mat)[iExt][ix][iz] += (*dataSliceVector[iGpu]->_mat)[iExt][ix][iz];
                }
			}
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateWemvaExtShotsGpu(iGpu);
	}

}
void wemvaExtShotsGpu::forwardWavefield(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double3DReg> data) {

	if (!add) data->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignalsVector.size() == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal = 0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1){constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vector of wemvaExtGpu objects (as many as the number of GPUs )
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2), data->getHyper()->getAxis(3))); // Hypercube for extended image
	std::vector<std::shared_ptr<double3DReg>> dataSliceVector;
	std::vector<std::shared_ptr<wemvaExtGpu>> wemvaExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended wemva object
		std::shared_ptr<wemvaExtGpu> wemvaExtGpuObject(new wemvaExtGpu(_vel, _par, _nGpu, iGpu));
		wemvaExtGpuObjectVector.push_back(wemvaExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			wemvaExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateWemvaExtShotsGpu(wemvaExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, wemvaExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu);

		// Create data slice (extended image) for this GPU number
		std::shared_ptr<SEP::double3DReg> dataSlice(new SEP::double3DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch wemva forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Change the wavefield flag
		if (iShot == _wavefieldShotNumber) {
			wemvaExtGpuObjectVector[iGpu]->setAllWavefields(1);
		} else {
			wemvaExtGpuObjectVector[iGpu]->setAllWavefields(0);
		}

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], _receiversSignalsVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], _receiversSignalsVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], _receiversSignalsVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], _receiversSignalsVector[iShot], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		wemvaExtGpuObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		wemvaExtGpuObjectVector[iGpu]->forward(false, model, dataSliceVector[iGpu]);

        // Stack data
		#pragma omp parallel for
		for (int iExt=0; iExt<hyperDataSlice->getAxis(3).n; iExt++){
			for (int ix=0; ix<hyperDataSlice->getAxis(2).n; ix++){
                for (int iz=0; iz<hyperDataSlice->getAxis(1).n; iz++){
                    (*data->_mat)[iExt][ix][iz] += (*dataSliceVector[iGpu]->_mat)[iExt][ix][iz];
                }
			}
		}

		// Get the wavefields
		if (iShot == _wavefieldShotNumber) {

			std::cout << "Saving wavefield from shot #" << iShot << ", computed by gpu #" << iGpu << std::endl;
			_srcWavefield = wemvaExtGpuObjectVector[iGpu]->getSrcWavefield();
			_secWavefield1 = wemvaExtGpuObjectVector[iGpu]->getSecWavefield1();
			_secWavefield2 = wemvaExtGpuObjectVector[iGpu]->getSecWavefield2();
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateWemvaExtShotsGpu(iGpu);
	}

}

/* Adjoint */
void wemvaExtShotsGpu::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double3DReg> data) const {

	if (!add) model->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignalsVector.size() == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal = 0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1){constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	std::vector<std::shared_ptr<wemvaExtGpu>> wemvaExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended wemva object
		std::shared_ptr<wemvaExtGpu> wemvaExtGpuObject(new wemvaExtGpu(_vel, _par, _nGpu, iGpu));
		wemvaExtGpuObjectVector.push_back(wemvaExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			wemvaExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device for that object
		allocateWemvaExtShotsGpu(wemvaExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, wemvaExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSlice->scale(0.0);
		modelSliceVector.push_back(modelSlice);
	}

	// Launch Born forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], _receiversSignalsVector[iShot], model, data);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], _receiversSignalsVector[iShot], model, data);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], _receiversSignalsVector[iShot], model, data);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], _receiversSignalsVector[iShot], model, data);
		}

		// Set GPU number for propagator object
		wemvaExtGpuObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		wemvaExtGpuObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], data);

	}

	// Stack models computed by each GPU
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		#pragma omp parallel for
		for (int ix=0; ix<model->getHyper()->getAxis(2).n; ix++){
			for (int iz=0; iz<model->getHyper()->getAxis(1).n; iz++){
				(*model->_mat)[ix][iz] += (*modelSliceVector[iGpu]->_mat)[ix][iz];
			}
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateWemvaExtShotsGpu(iGpu);
	}
}
void wemvaExtShotsGpu::adjointWavefield(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double3DReg> data) {

	if (!add) model->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (_sourcesSignalsVector.size() == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal = 0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1){constantRecGeom=1;}
	else {constantRecGeom=0;}

	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	std::vector<std::shared_ptr<wemvaExtGpu>> wemvaExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended wemva object
		std::shared_ptr<wemvaExtGpu> wemvaExtGpuObject(new wemvaExtGpu(_vel, _par, _nGpu, iGpu));
		wemvaExtGpuObjectVector.push_back(wemvaExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			wemvaExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device for that object
		allocateWemvaExtShotsGpu(wemvaExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, wemvaExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSlice->scale(0.0);
		modelSliceVector.push_back(modelSlice);

	}

	// Launch wemva adjoint
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Change the wavefield flag
		if (iShot == _wavefieldShotNumber) {
			wemvaExtGpuObjectVector[iGpu]->setWavefield(1);
		} else {
			wemvaExtGpuObjectVector[iGpu]->setWavefield(0);
		}

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], _receiversSignalsVector[iShot], model, data);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], _receiversSignalsVector[iShot], model, data);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], _receiversSignalsVector[iShot], model, data);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			wemvaExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], _receiversSignalsVector[iShot], model, data);
		}

		// Set GPU number for propagator object
		wemvaExtGpuObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		wemvaExtGpuObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], data);

		// Get the wavefields
		if (iShot == _wavefieldShotNumber) {
			std::cout << "Finished propagation of shot #" << iShot << ", computed by gpu #" << iGpu << " - saving wavefield" << std::endl;
			_srcWavefield = wemvaExtGpuObjectVector[iGpu]->getSrcWavefield();
			_secWavefield1 = wemvaExtGpuObjectVector[iGpu]->getSecWavefield1();
			_secWavefield2 = wemvaExtGpuObjectVector[iGpu]->getSecWavefield2(); // Receiver wavefield
		}
	}

	// Stack models computed by each GPU
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		#pragma omp parallel for
		for (int ix=0; ix<model->getHyper()->getAxis(2).n; ix++){
			for (int iz=0; iz<model->getHyper()->getAxis(1).n; iz++){
				(*model->_mat)[ix][iz] += (*modelSliceVector[iGpu]->_mat)[ix][iz];
			}
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateWemvaExtShotsGpu(iGpu);
	}

}

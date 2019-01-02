#include <vector>
#include <omp.h>
#include "BornShotsGpu.h"
#include "BornGpu.h"

BornShotsGpu::BornShotsGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<SEP::double2DReg>> sourcesSignalsVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector){

	// Setup parameters
	_par = par;
	_vel = vel;
	_nShot = par->getInt("nShot");
	_nGpu = par->getInt("nGpu");
	_info = par->getInt("info", 0);
	_deviceNumberInfo = par->getInt("deviceNumberInfo", 0);
	assert(getGpuInfo(_nGpu, _info, _deviceNumberInfo)); // Get info on GPU cluster and check that there are enough available GPUs
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_wavefieldShotNumber = _par->getInt("wavefieldShotNumber", 0);
	_sourcesVector = sourcesVector;
	_receiversVector = receiversVector;
	_sourcesSignalsVector = sourcesSignalsVector;

}
void BornShotsGpu::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double3DReg> data) const {

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

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<BornGpu>> BornObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		std::shared_ptr<BornGpu> BornGpuObject(new BornGpu(_vel, _par, _nGpu, iGpu));
		BornObjectVector.push_back(BornGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			BornGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateBornShotsGpu(BornObjectVector[iGpu]->getFdParam()->_vel2Dtw2, BornObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch Born forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		BornObjectVector[iGpu]->forward(false, model, dataSliceVector[iGpu]);

		// Store dataSlice into data
		#pragma omp parallel for
		for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
			for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
				(*data->_mat)[iShot][iReceiver][its] += (*dataSliceVector[iGpu]->_mat)[iReceiver][its];
			}
		}

	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateBornShotsGpu(iGpu);
	}

}
void BornShotsGpu::forwardWavefield(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double3DReg> data) {

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

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<BornGpu>> BornObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		std::shared_ptr<BornGpu> BornGpuObject(new BornGpu(_vel, _par, _nGpu, iGpu));
		BornObjectVector.push_back(BornGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			BornGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateBornShotsGpu(BornObjectVector[iGpu]->getFdParam()->_vel2Dtw2, BornObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu);
		BornObjectVector[iGpu]->setAllWavefields(0); // By default, do not record the scattered wavefields

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch Born forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Change the wavefield flag
		if (iShot == _wavefieldShotNumber) {
			std::cout << "Allocating wavefields" << std::endl;
			BornObjectVector[iGpu]->setAllWavefields(1);
			std::cout << "Done allocating wavefields" << std::endl;
		} else {
			BornObjectVector[iGpu]->setAllWavefields(0);
		}

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		BornObjectVector[iGpu]->forward(false, model, dataSliceVector[iGpu]);
		// Store dataSlice into data
		#pragma omp parallel for
		for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
			for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
				(*data->_mat)[iShot][iReceiver][its] += (*dataSliceVector[iGpu]->_mat)[iReceiver][its];
			}
		}

		// Get the wavefields
		if (iShot == _wavefieldShotNumber) {
			std::cout << "Saving wavefield from shot # " << iShot << ", computed by gpu # " << iGpu << std::endl;
			_srcWavefield = BornObjectVector[iGpu]->getSrcWavefield();
			_secWavefield = BornObjectVector[iGpu]->getSecWavefield();
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateBornShotsGpu(iGpu);
	}

}
void BornShotsGpu::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double3DReg> data) const {

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
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector, dataSliceVector;
	std::vector<std::shared_ptr<BornGpu>> BornObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		std::shared_ptr<BornGpu> BornGpuObject(new BornGpu(_vel, _par, _nGpu, iGpu));
		BornObjectVector.push_back(BornGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			BornGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device for that object
		allocateBornShotsGpu(BornObjectVector[iGpu]->getFdParam()->_vel2Dtw2, BornObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSlice->scale(0.0); // Check that
		modelSliceVector.push_back(modelSlice);


		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);
	}

	// Launch Born forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		BornObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);

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
		deallocateBornShotsGpu(iGpu);
	}

}
void BornShotsGpu::adjointWavefield(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double3DReg> data) {

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
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector, dataSliceVector;
	std::vector<std::shared_ptr<BornGpu>> BornObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create Born object
		std::shared_ptr<BornGpu> BornGpuObject(new BornGpu(_vel, _par, _nGpu, iGpu));
		BornObjectVector.push_back(BornGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			BornGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device for that object
		allocateBornShotsGpu(BornObjectVector[iGpu]->getFdParam()->_vel2Dtw2, BornObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu);
		BornObjectVector[iGpu]->setAllWavefields(0); // By default, do not record the scattered wavefields

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSlice->scale(0.0); // Check if we really need that?
		modelSliceVector.push_back(modelSlice);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);
	}

	// Launch Born forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Change the wavefield flag
		if (iShot == _wavefieldShotNumber) {
			BornObjectVector[iGpu]->setAllWavefields(1);
		} else {
			BornObjectVector[iGpu]->setAllWavefields(0);
		}

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			BornObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		BornObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);

		// Get the wavefields
		if (iShot == _wavefieldShotNumber) {
			std::cout << "Finished propagation of shot# " << iShot << ", computed by gpu# " << iGpu << " - saving wavefields" << std::endl;
			_srcWavefield = BornObjectVector[iGpu]->getSrcWavefield();
			_secWavefield = BornObjectVector[iGpu]->getSecWavefield();
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
		deallocateBornShotsGpu(iGpu);
	}

}

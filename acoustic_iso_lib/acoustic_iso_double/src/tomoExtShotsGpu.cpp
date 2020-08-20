#include <vector>
#include <omp.h>
#include "tomoExtShotsGpu.h"
#include "tomoExtGpu.h"

/* Constructor */
tomoExtShotsGpu::tomoExtShotsGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<SEP::double2DReg>> sourcesSignalsVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector, std::shared_ptr<SEP::double3DReg> reflectivityExt) {

	// Setup parameters
	_par = par;
	_vel = vel;
	_nShot = par->getInt("nShot");
	createGpuIdList();
	_info = par->getInt("info", 0);
	_deviceNumberInfo = par->getInt("deviceNumberInfo", _gpuList[0]);
	if( not getGpuInfo(_gpuList, _info, _deviceNumberInfo)){
		throw std::runtime_error("");
	}; // Get info on GPU cluster and check that there are enough available GPUs
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_wavefieldShotNumber = _par->getInt("wavefieldShotNumber", 0);
	_sourcesVector = sourcesVector;
	_receiversVector = receiversVector;
	_sourcesSignalsVector = sourcesSignalsVector;
	_reflectivityExt = reflectivityExt;

}

void tomoExtShotsGpu::createGpuIdList(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);
	std::vector<int> dummyVector;
 	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){std::cout << "**** ERROR: Please provide a list of GPUs to be used ****" << std::endl; throw std::runtime_error("");}

	// If user does not provide a valid list but provides nGpu -> use id: 0,...,nGpu-1
	if (_nGpu>0 && _gpuList[0]<0){
		_gpuList.clear();
		for (int iGpu=0; iGpu<_nGpu; iGpu++){
			_gpuList.push_back(iGpu);
		}
	}

	// If the user provides a list -> use that list and ignore nGpu for the parfile
	if (_gpuList[0]>=0){
		_nGpu = _gpuList.size();
		sort(_gpuList.begin(), _gpuList.end());
		std::vector<int>::iterator it = std::unique(_gpuList.begin(), _gpuList.end());
		bool isUnique = (it==_gpuList.end());
		if (isUnique==0) {
			std::cout << "**** ERROR: Please make sure there are no duplicates in the list ****" << std::endl; throw std::runtime_error("");
		}
	}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}

/* Forward */
void tomoExtShotsGpu::forward(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double3DReg> data) const {

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
	std::vector<std::shared_ptr<tomoExtGpu>> tomoExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended tomo object
		std::shared_ptr<tomoExtGpu> tomoExtGpuObject(new tomoExtGpu(_vel, _par, _reflectivityExt, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		tomoExtGpuObjectVector.push_back(tomoExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			tomoExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateTomoExtShotsGpu(tomoExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, tomoExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

 	// Launch Tomo forward
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		tomoExtGpuObjectVector[iGpu]->setGpuNumber(iGpu, iGpuId);

		// Launch modeling
		tomoExtGpuObjectVector[iGpu]->forward(false, model, dataSliceVector[iGpu]);

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
		deallocateTomoExtShotsGpu(iGpu, _gpuList[iGpu]);
	}

}
void tomoExtShotsGpu::forwardWavefield(const bool add, const std::shared_ptr<double2DReg> model, std::shared_ptr<double3DReg> data) {

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
	std::vector<std::shared_ptr<tomoExtGpu>> tomoExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended tomo object
		std::shared_ptr<tomoExtGpu> tomoExtGpuObject(new tomoExtGpu(_vel, _par, _reflectivityExt, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		tomoExtGpuObjectVector.push_back(tomoExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			tomoExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateTomoExtShotsGpu(tomoExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, tomoExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch tomo forward
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Change the wavefield flag
		if (iShot == _wavefieldShotNumber) {
			tomoExtGpuObjectVector[iGpu]->setAllWavefields(1);
		} else {
			tomoExtGpuObjectVector[iGpu]->setAllWavefields(0);
		}

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		tomoExtGpuObjectVector[iGpu]->setGpuNumber(iGpu, iGpuId);

		// Launch modeling
		tomoExtGpuObjectVector[iGpu]->forward(false, model, dataSliceVector[iGpu]);

		// Store dataSlice into data
		#pragma omp parallel for
		for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
			for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
				(*data->_mat)[iShot][iReceiver][its] += (*dataSliceVector[iGpu]->_mat)[iReceiver][its];
			}
		}

		// Get the wavefields
		if (iShot == _wavefieldShotNumber) {

			std::cout << "Saving wavefield from shot #" << iShot << ", computed by gpu #" << iGpu << std::endl;
			_srcWavefield = tomoExtGpuObjectVector[iGpu]->getSrcWavefield(); // Source wavefield
			_secWavefield1 = tomoExtGpuObjectVector[iGpu]->getSecWavefield1(); // First scattered wavefield
			_secWavefield2 = tomoExtGpuObjectVector[iGpu]->getSecWavefield2(); // Second scattered wavefield
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateTomoExtShotsGpu(iGpu, _gpuList[iGpu]);
	}
}

/* Adjoint */
void tomoExtShotsGpu::adjoint(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double3DReg> data) const {

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
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	std::vector<std::shared_ptr<tomoExtGpu>> tomoExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended Born object
		std::shared_ptr<tomoExtGpu> tomoExtGpuObject(new tomoExtGpu(_vel, _par, _reflectivityExt, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		tomoExtGpuObjectVector.push_back(tomoExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			tomoExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device for that object
		allocateTomoExtShotsGpu(tomoExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, tomoExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSlice->scale(0.0);
		modelSliceVector.push_back(modelSlice);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);
	}

	// Launch Born forward
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		tomoExtGpuObjectVector[iGpu]->setGpuNumber(iGpu, iGpuId);

		// Launch modeling
		tomoExtGpuObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);

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
		deallocateTomoExtShotsGpu(iGpu, _gpuList[iGpu]);
	}
}
void tomoExtShotsGpu::adjointWavefield(const bool add, std::shared_ptr<double2DReg> model, const std::shared_ptr<double3DReg> data) {

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
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	std::vector<std::shared_ptr<tomoExtGpu>> tomoExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended Born object
		std::shared_ptr<tomoExtGpu> tomoExtGpuObject(new tomoExtGpu(_vel, _par, _reflectivityExt, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		tomoExtGpuObjectVector.push_back(tomoExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			tomoExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device for that object
		allocateTomoExtShotsGpu(tomoExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, tomoExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSlice->scale(0.0);
		modelSliceVector.push_back(modelSlice);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch tomo adjoint
	#pragma omp parallel for schedule(dynamic,1) num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Change the wavefield flag
		if (iShot == _wavefieldShotNumber) {
			tomoExtGpuObjectVector[iGpu]->setWavefield(1);
		} else {
			tomoExtGpuObjectVector[iGpu]->setWavefield(0);
		}

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			tomoExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		tomoExtGpuObjectVector[iGpu]->setGpuNumber(iGpu, iGpuId);

		// Launch modeling
		tomoExtGpuObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);

		// Get the wavefields
		if (iShot == _wavefieldShotNumber) {
			std::cout << "Finished propagation of shot #" << iShot << ", computed by gpu #" << iGpu << " - saving wavefield" << std::endl;
			_srcWavefield = tomoExtGpuObjectVector[iGpu]->getSrcWavefield();
			_secWavefield1 = tomoExtGpuObjectVector[iGpu]->getSecWavefield1(); // Intermediate scattered wavefield
			_secWavefield2 = tomoExtGpuObjectVector[iGpu]->getSecWavefield2(); // Receiver wavefield
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
		deallocateTomoExtShotsGpu(iGpu, _gpuList[iGpu]);
	}
}

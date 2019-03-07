#include <vector>
#include <omp.h>
#include "BornExtShotsGpu.h"
#include "BornExtGpu.h"

BornExtShotsGpu::BornExtShotsGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<SEP::double2DReg>> sourcesSignalsVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector){

	// Setup parameters
	_par = par;
	_vel = vel;
	_nShot = par->getInt("nShot");
	createGpuIdList();
	_info = par->getInt("info", 0);
	_deviceNumberInfo = par->getInt("deviceNumberInfo", _gpuList[0]);
	assert(getGpuInfo(_gpuList, _info, _deviceNumberInfo)); // Get info on GPU cluster and check that there are enough available GPUs
	_saveWavefield = _par->getInt("saveWavefield", 0);
	_wavefieldShotNumber = _par->getInt("wavefieldShotNumber", 0);
	_sourcesVector = sourcesVector;
	_receiversVector = receiversVector;
	_sourcesSignalsVector = sourcesSignalsVector;

}

void BornExtShotsGpu::createGpuIdList(){

	// Setup Gpu numbers
	_nGpu = _par->getInt("nGpu", -1);
	std::vector<int> dummyVector;
 	dummyVector.push_back(-1);
	_gpuList = _par->getInts("iGpu", dummyVector);

	// If the user does not provide nGpu > 0 or a valid list -> break
	if (_nGpu <= 0 && _gpuList[0]<0){std::cout << "**** ERROR: Please provide a list of GPUs to be used ****" << std::endl; assert(1==2);}

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
			std::cout << "**** ERROR: Please make sure there are no duplicates in the list ****" << std::endl; assert(1==2);
		}
	}

	// Allocation of arrays of arrays will be done by the gpu # _gpuList[0]
	_iGpuAlloc = _gpuList[0];
}

void BornExtShotsGpu::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

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
	std::vector<std::shared_ptr<BornExtGpu>> BornExtObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended Born object
		std::shared_ptr<BornExtGpu> BornExtGpuObject(new BornExtGpu(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornExtObjectVector.push_back(BornExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			BornExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateBornExtShotsGpu(BornExtObjectVector[iGpu]->getFdParam()->_vel2Dtw2, BornExtObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);
	}

	// Launch Born forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			BornExtObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			BornExtObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornExtObjectVector[iGpu]->setGpuNumber(iGpu, iGpuId);

		// Launch modeling
		BornExtObjectVector[iGpu]->forward(false, model, dataSliceVector[iGpu]);

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
		deallocateBornExtShotsGpu(iGpu, _gpuList[iGpu]);
	}

}
void BornExtShotsGpu::forwardWavefield(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) {

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
	std::vector<std::shared_ptr<BornExtGpu>> BornExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended Born object
		std::shared_ptr<BornExtGpu> BornExtGpuObject(new BornExtGpu(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornExtGpuObjectVector.push_back(BornExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			BornExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateBornExtShotsGpu(BornExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, BornExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch Born forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Change the wavefield flag
		if (iShot == _wavefieldShotNumber) {
			std::cout << "Allocating wavefields" << std::endl;
			BornExtGpuObjectVector[iGpu]->setAllWavefields(1);
			std::cout << "Done allocating wavefields" << std::endl;
		} else {
			BornExtGpuObjectVector[iGpu]->setAllWavefields(0);
		}

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			BornExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			BornExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornExtGpuObjectVector[iGpu]->setGpuNumber(iGpu, iGpuId);

		// Launch modeling
		BornExtGpuObjectVector[iGpu]->forward(false, model, dataSliceVector[iGpu]);

		// Store dataSlice into data
		#pragma omp parallel for
		for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
			for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
				(*data->_mat)[iShot][iReceiver][its] += (*dataSliceVector[iGpu]->_mat)[iReceiver][its];
			}
		}

		// Get the wavefields
		if (iShot == _wavefieldShotNumber) {
			std::cout << "Finished propagation of shot# " << iShot << ", computed by gpu# " << iGpu << " - saving wavefield" << std::endl;
			_srcWavefield = BornExtGpuObjectVector[iGpu]->getSrcWavefield();
			_secWavefield = BornExtGpuObjectVector[iGpu]->getSecWavefield();
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateBornExtShotsGpu(iGpu, _gpuList[iGpu]);
	}

}
void BornExtShotsGpu::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const {

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
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<double3DReg>> modelSliceVector;
	std::vector<std::shared_ptr<BornExtGpu>> BornExtGpuObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended Born object
		std::shared_ptr<BornExtGpu> BornExtGpuObject(new BornExtGpu(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornExtGpuObjectVector.push_back(BornExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			BornExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device for that object
		allocateBornExtShotsGpu(BornExtGpuObjectVector[iGpu]->getFdParam()->_vel2Dtw2, BornExtGpuObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Model slice
		std::shared_ptr<SEP::double3DReg> modelSlice(new SEP::double3DReg(hyperModelSlice));
		modelSlice->scale(0.0);
		modelSliceVector.push_back(modelSlice);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);
	}

	// Launch Born forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			BornExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			BornExtGpuObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornExtGpuObjectVector[iGpu]->setGpuNumber(iGpu, iGpuId);

		// Launch modeling
		BornExtGpuObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);

	}

	// Stack models computed by each GPU
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		#pragma omp parallel for
		for (int iExt=0; iExt<model->getHyper()->getAxis(3).n; iExt++){
			for (int ix=0; ix<model->getHyper()->getAxis(2).n; ix++){
				for (int iz=0; iz<model->getHyper()->getAxis(1).n; iz++){
					(*model->_mat)[iExt][ix][iz] += (*modelSliceVector[iGpu]->_mat)[iExt][ix][iz];
				}
			}
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateBornExtShotsGpu(iGpu, _gpuList[iGpu]);
	}
}
void BornExtShotsGpu::adjointWavefield(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) {

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
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2), model->getHyper()->getAxis(3)));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<double3DReg>> modelSliceVector;
	std::vector<std::shared_ptr<BornExtGpu>> BornExtObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Create extended Born object
		std::shared_ptr<BornExtGpu> BornExtGpuObject(new BornExtGpu(_vel, _par, _nGpu, iGpu, _gpuList[iGpu], _iGpuAlloc));
		BornExtObjectVector.push_back(BornExtGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (_gpuList[iGpu] == _deviceNumberInfo) ){
			BornExtGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device for that object
		allocateBornExtShotsGpu(BornExtObjectVector[iGpu]->getFdParam()->_vel2Dtw2, BornExtObjectVector[iGpu]->getFdParam()->_reflectivityScale, iGpu, _gpuList[iGpu]);

		// Model slice
		std::shared_ptr<SEP::double3DReg> modelSlice(new SEP::double3DReg(hyperModelSlice));
		modelSlice->scale(0.0);
		modelSliceVector.push_back(modelSlice);

		// Create data slice for this GPU number
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch Born adjoint
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();
		int iGpuId = _gpuList[iGpu];

		// Change the wavefield flag
		if (iShot == _wavefieldShotNumber) {
			BornExtObjectVector[iGpu]->setAllWavefields(1);
		} else {
			BornExtObjectVector[iGpu]->setAllWavefields(0);
		}

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if ( (constantRecGeom == 1) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 1) && (constantSrcSignal == 0) ) {
			BornExtObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 1) ) {
			BornExtObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[0], _receiversVector[iShot], model, dataSliceVector[iGpu]);
		}
		if ( (constantRecGeom == 0) && (constantSrcSignal == 0) ) {
			BornExtObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _sourcesSignalsVector[iShot], _receiversVector[0], model, dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		BornExtObjectVector[iGpu]->setGpuNumber(iGpu, iGpuId);

		// Launch modeling
		BornExtObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);

		// Get the wavefields
		if (iShot == _wavefieldShotNumber) {
			std::cout << "Finished propagation of shot# " << iShot << ", computed by gpu# " << iGpu << " - saving wavefield" << std::endl;
			_srcWavefield = BornExtObjectVector[iGpu]->getSrcWavefield();
			_secWavefield = BornExtObjectVector[iGpu]->getSecWavefield();
		}
	}

	// Stack models computed by each GPU
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		#pragma omp parallel for
		for (int iExt=0; iExt<model->getHyper()->getAxis(3).n; iExt++){
			for (int ix=0; ix<model->getHyper()->getAxis(2).n; ix++){
				for (int iz=0; iz<model->getHyper()->getAxis(1).n; iz++){
					(*model->_mat)[iExt][ix][iz] += (*modelSliceVector[iGpu]->_mat)[iExt][ix][iz];
				}
			}
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateBornExtShotsGpu(iGpu, _gpuList[iGpu]);
	}
}

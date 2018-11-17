#include <vector>
#include <omp.h>
#include "nonlinearPropShotsGpu.h"
#include "nonlinearPropGpu.h"

// Cosntructor
nonlinearPropShotsGpu::nonlinearPropShotsGpu(std::shared_ptr<SEP::double2DReg> vel, std::shared_ptr<paramObj> par, std::vector<std::shared_ptr<deviceGpu>> sourcesVector, std::vector<std::shared_ptr<deviceGpu>> receiversVector) {

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
	if (_info == 1){std::cout << "Saving wavefield(s) for shot # " << _wavefieldShotNumber << std::endl;}
	_sourcesVector = sourcesVector;
	_receiversVector = receiversVector;

}

// Forward
void nonlinearPropShotsGpu::forward(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) const {

	if (!add) data->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (model->getHyper()->getAxis(3).n == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal=0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1) {constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2)));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<nonlinearPropGpu>> propObjectVector;

	// Initialization for each GPU:
	// (1) Creation of vector of objects, model, and data.
	// (2) Memory allocation on GPU
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Nonlinear propagator object
		std::shared_ptr<nonlinearPropGpu> propGpuObject(new nonlinearPropGpu(_vel, _par, _nGpu, iGpu));
		propObjectVector.push_back(propGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			propGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateNonlinearGpu(propObjectVector[iGpu]->getFdParam()->_vel2Dtw2, iGpu);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSliceVector.push_back(modelSlice);

		// Data slice
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch nonlinear forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Copy model slice
		if(constantSrcSignal == 1) {
			memcpy(modelSliceVector[iGpu]->getVals(), &(model->getVals()[0]), sizeof(double)*hyperModelSlice->getAxis(1).n*hyperModelSlice->getAxis(2).n);
		} else {
			memcpy(modelSliceVector[iGpu]->getVals(), &(model->getVals()[iShot*hyperModelSlice->getAxis(1).n*hyperModelSlice->getAxis(2).n]), sizeof(double)*hyperModelSlice->getAxis(1).n*hyperModelSlice->getAxis(2).n);
		}

		// Set acquisition geometry
		if (constantRecGeom == 1) {
			propObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _receiversVector[0], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		} else {
			propObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _receiversVector[iShot], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		propObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		propObjectVector[iGpu]->forward(false, modelSliceVector[iGpu], dataSliceVector[iGpu]);

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
		deallocateNonlinearGpu(iGpu);
	}

}
void nonlinearPropShotsGpu::forwardWavefield(const bool add, const std::shared_ptr<double3DReg> model, std::shared_ptr<double3DReg> data) {

	if (!add) data->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (model->getHyper()->getAxis(3).n == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal = 0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1){constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2)));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<nonlinearPropGpu>> propObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Nonlinear propagator object
		std::shared_ptr<nonlinearPropGpu> propGpuObject(new nonlinearPropGpu(_vel, _par, _nGpu, iGpu));
		propObjectVector.push_back(propGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			propGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateNonlinearGpu(propObjectVector[iGpu]->getFdParam()->_vel2Dtw2, iGpu);
		propObjectVector[iGpu]->setAllWavefields(0); // By default, do not record the scattered wavefields

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSliceVector.push_back(modelSlice);

		// Data slice
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch nonlinear forward
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Change the wavefield flag
		if (iShot == _wavefieldShotNumber) {
			propObjectVector[iGpu]->setAllWavefields(1);
		} else {
			propObjectVector[iGpu]->setAllWavefields(0);
		}

		// Copy model slice
		if(constantSrcSignal == 1){
			memcpy(modelSliceVector[iGpu]->getVals(), &(model->getVals()[0]), sizeof(double)*hyperModelSlice->getAxis(1).n*hyperModelSlice->getAxis(2).n);
		} else {
			memcpy(modelSliceVector[iGpu]->getVals(), &(model->getVals()[iShot*hyperModelSlice->getAxis(1).n*hyperModelSlice->getAxis(2).n]), sizeof(double)*hyperModelSlice->getAxis(1).n*hyperModelSlice->getAxis(2).n);
		}

		// Set acquisition geometry
		if(constantRecGeom == 1) {
			propObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _receiversVector[0], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		} else {
			propObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _receiversVector[iShot], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		propObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		propObjectVector[iGpu]->forward(false, modelSliceVector[iGpu], dataSliceVector[iGpu]);

		// Store dataSlice into data
		#pragma omp parallel for
		for (int iReceiver=0; iReceiver<hyperDataSlice->getAxis(2).n; iReceiver++){
			for (int its=0; its<hyperDataSlice->getAxis(1).n; its++){
				(*data->_mat)[iShot][iReceiver][its] += (*dataSliceVector[iGpu]->_mat)[iReceiver][its];
			}
		}

		// Get the wavefield
		if (iShot == _wavefieldShotNumber) {
			_wavefield = propObjectVector[iGpu]->getWavefield();

		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateNonlinearGpu(iGpu);
	}

}

// Adjoint
void nonlinearPropShotsGpu::adjoint(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) const {

	if (!add) model->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (model->getHyper()->getAxis(3).n == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal = 0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1){constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2)));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<nonlinearPropGpu>> propObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Nonlinear propagator object
		std::shared_ptr<nonlinearPropGpu> propGpuObject(new nonlinearPropGpu(_vel, _par, _nGpu, iGpu));
		propObjectVector.push_back(propGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			propGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateNonlinearGpu(propObjectVector[iGpu]->getFdParam()->_vel2Dtw2, iGpu);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSliceVector.push_back(modelSlice);
		modelSliceVector[iGpu]->scale(0.0); // Initialize each model slice to zero

		// Data slice
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch nonlinear adjoint
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if(constantRecGeom == 1) {
			propObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _receiversVector[0], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		} else {
			propObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _receiversVector[iShot], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		propObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		if (constantSrcSignal == 1){
			// Stack all shots for the same iGpu (and we need to re-stack everything at the end)
			propObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);
		} else {
			// Copy the shot into model slice --> Is there a way to parallelize this?
			propObjectVector[iGpu]->adjoint(false, modelSliceVector[iGpu], dataSliceVector[iGpu]);
			#pragma omp parallel for
			for (int iSource=0; iSource<hyperModelSlice->getAxis(2).n; iSource++){
				for (int its=0; its<hyperModelSlice->getAxis(1).n; its++){
					(*model->_mat)[iShot][iSource][its] += (*modelSliceVector[iGpu]->_mat)[iSource][its];
				}
			}
		}
	}

	// If same sources for all shots, stack all shots from all iGpus
	if (constantSrcSignal == 1){
		for (int iSource=0; iSource<hyperModelSlice->getAxis(2).n; iSource++){
			#pragma omp parallel for
			for (int its=0; its<hyperModelSlice->getAxis(1).n; its++){
				for (int iGpu=0; iGpu<_nGpu; iGpu++){
					(*model->_mat)[0][iSource][its]	+= (*modelSliceVector[iGpu]->_mat)[iSource][its];
				}
			}
		}
	}

	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateNonlinearGpu(iGpu);
	}

}
void nonlinearPropShotsGpu::adjointWavefield(const bool add, std::shared_ptr<double3DReg> model, const std::shared_ptr<double3DReg> data) {

	if (!add) model->scale(0.0);

	// Variable declaration
	int omp_get_thread_num();
	int constantSrcSignal, constantRecGeom;

	// Check whether we use the same source signals for all shots
	if (model->getHyper()->getAxis(3).n == 1) {constantSrcSignal = 1;}
	else {constantSrcSignal = 0;}

	// Check if we have constant receiver geometry
	if (_receiversVector.size() == 1){constantRecGeom=1;}
	else {constantRecGeom=0;}

	// Create vectors for each GPU
	std::shared_ptr<SEP::hypercube> hyperModelSlice(new hypercube(model->getHyper()->getAxis(1), model->getHyper()->getAxis(2)));
	std::shared_ptr<SEP::hypercube> hyperDataSlice(new hypercube(data->getHyper()->getAxis(1), data->getHyper()->getAxis(2)));
	std::vector<std::shared_ptr<double2DReg>> modelSliceVector;
	std::vector<std::shared_ptr<double2DReg>> dataSliceVector;
	std::vector<std::shared_ptr<nonlinearPropGpu>> propObjectVector;

	// Loop over GPUs
	for (int iGpu=0; iGpu<_nGpu; iGpu++){

		// Nonlinear propagator object
		std::shared_ptr<nonlinearPropGpu> propGpuObject(new nonlinearPropGpu(_vel, _par, _nGpu, iGpu));
		propObjectVector.push_back(propGpuObject);

		// Display finite-difference parameters info
		if ( (_info == 1) && (iGpu == _deviceNumberInfo) ){
			propGpuObject->getFdParam()->getInfo();
		}

		// Allocate memory on device
		allocateNonlinearGpu(propObjectVector[iGpu]->getFdParam()->_vel2Dtw2, iGpu);
		propObjectVector[iGpu]->setWavefield(0);

		// Model slice
		std::shared_ptr<SEP::double2DReg> modelSlice(new SEP::double2DReg(hyperModelSlice));
		modelSliceVector.push_back(modelSlice);
		modelSliceVector[iGpu]->scale(0.0); // Initialize each model slice to zero

		// Data slice
		std::shared_ptr<SEP::double2DReg> dataSlice(new SEP::double2DReg(hyperDataSlice));
		dataSliceVector.push_back(dataSlice);

	}

	// Launch nonlinear adjoint
	#pragma omp parallel for num_threads(_nGpu)
	for (int iShot=0; iShot<_nShot; iShot++){

		int iGpu = omp_get_thread_num();

		// Change the saveWavefield flag
		if (iShot == _wavefieldShotNumber) { propObjectVector[iGpu]->setAllWavefields(1);}

		// Copy data slice
		memcpy(dataSliceVector[iGpu]->getVals(), &(data->getVals()[iShot*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n]), sizeof(double)*hyperDataSlice->getAxis(1).n*hyperDataSlice->getAxis(2).n);

		// Set acquisition geometry
		if(constantRecGeom == 1) {
			propObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _receiversVector[0], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		} else {
			propObjectVector[iGpu]->setAcquisition(_sourcesVector[iShot], _receiversVector[iShot], modelSliceVector[iGpu], dataSliceVector[iGpu]);
		}

		// Set GPU number for propagator object
		propObjectVector[iGpu]->setGpuNumber(iGpu);

		// Launch modeling
		if (constantSrcSignal == 1){

			// Stack all shots for the same iGpu (and we need to re-stack everything at the end)
			propObjectVector[iGpu]->adjoint(true, modelSliceVector[iGpu], dataSliceVector[iGpu]);

		} else {

			// Copy the shot into model slice --> Is there a way to parallelize this?
			propObjectVector[iGpu]->adjoint(false, modelSliceVector[iGpu], dataSliceVector[iGpu]);
			#pragma omp parallel for
			for (int iSource=0; iSource<hyperModelSlice->getAxis(2).n; iSource++){
				for (int its=0; its<hyperModelSlice->getAxis(1).n; its++){
					(*model->_mat)[iShot][iSource][its] += (*modelSliceVector[iGpu]->_mat)[iSource][its];
				}

			}
		}

		// Get the wavefield
		if (iShot == _wavefieldShotNumber) {
			_wavefield = propObjectVector[iGpu]->getWavefield();
		}
	}

	// If same sources for all shots, stack all shots from all iGpus
	if (constantSrcSignal == 1){
		for (int iSource=0; iSource<hyperModelSlice->getAxis(2).n; iSource++){
			#pragma omp parallel for
			for (int its=0; its<hyperModelSlice->getAxis(1).n; its++){
				for (int iGpu=0; iGpu<_nGpu; iGpu++){
					(*model->_mat)[0][iSource][its]	+= (*modelSliceVector[iGpu]->_mat)[iSource][its];
				}
			}
		}
	}
	// Deallocate memory on device
	for (int iGpu=0; iGpu<_nGpu; iGpu++){
		deallocateNonlinearGpu(iGpu);
	}

}
